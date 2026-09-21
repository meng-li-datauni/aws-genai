import os

import aws_cdk as cdk
from aws_cdk import Stack, aws_bedrock as bedrock, aws_iam as iam, aws_lambda as _lambda
from aws_cdk import aws_apigatewayv2 as apigw, aws_cloudfront as cloudfront, aws_cognito as cognito
from aws_cdk import aws_cloudfront_origins as origins, aws_s3 as s3, aws_s3_deployment as s3deploy
from aws_cdk import aws_s3vectors as s3v
from aws_cdk.aws_apigatewayv2_authorizers import HttpJwtAuthorizer
from aws_cdk.aws_apigatewayv2_integrations import HttpLambdaIntegration
from constructs import Construct

WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 1024


class RagStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Source documents (upload ai-professional-01.pdf here)
        docs = s3.Bucket(
            self, "Docs",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )

        # Vector store: S3 Vectors
        vector_bucket = s3v.CfnVectorBucket(self, "VectorBucket")
        index = s3v.CfnIndex(
            self, "VectorIndex",
            vector_bucket_arn=vector_bucket.attr_vector_bucket_arn,
            data_type="float32",
            dimension=EMBED_DIMENSIONS,
            distance_metric="cosine",
            # Bedrock stores chunk text in metadata; keep it non-filterable (size limit)
            metadata_configuration=s3v.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
            ),
        )

        embed_model_arn = f"arn:aws:bedrock:{self.region}::foundation-model/{EMBED_MODEL_ID}"

        # Role the Knowledge Base assumes to read S3, embed, and write vectors
        kb_role = iam.Role(
            self, "KbRole", assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com")
        )
        docs.grant_read(kb_role)
        kb_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"], resources=[embed_model_arn]))
        kb_role.add_to_policy(iam.PolicyStatement(
            actions=["s3vectors:GetIndex", "s3vectors:QueryVectors", "s3vectors:PutVectors",
                     "s3vectors:GetVectors", "s3vectors:DeleteVectors"],
            resources=[index.attr_index_arn]))

        kb = bedrock.CfnKnowledgeBase(
            self, "KnowledgeBase",
            name="aip-c01-guide-kb",
            role_arn=kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embed_model_arn,
                    embedding_model_configuration=bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                        bedrock_embedding_model_configuration=bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                            dimensions=EMBED_DIMENSIONS, embedding_data_type="FLOAT32")),
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="S3_VECTORS",
                s3_vectors_configuration=bedrock.CfnKnowledgeBase.S3VectorsConfigurationProperty(
                    index_arn=index.attr_index_arn),
            ),
        )
        kb.node.add_dependency(kb_role)

        data_source = bedrock.CfnDataSource(
            self, "DataSource",
            knowledge_base_id=kb.attr_knowledge_base_id,
            name="s3-docs",
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=docs.bucket_arn),
            ),
        )

        # Query API: Lambda + Function URL (IAM-authenticated)
        fn = _lambda.Function(
            self, "QueryFn",
            runtime=_lambda.Runtime.PYTHON_3_13,
            handler="lambda_handler.handler",
            code=_lambda.Code.from_asset("lambda_build"),  # run build_lambda.sh first
            timeout=cdk.Duration.minutes(2),
            memory_size=512,
            environment={"KNOWLEDGE_BASE_ID": kb.attr_knowledge_base_id},  # Lambda sets AWS_REGION itself
        )
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:Retrieve"], resources=[kb.attr_knowledge_base_arn]))
        # Claude via Bedrock; tighten to your model/inference-profile ARNs for production
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=["*"]))
        url = fn.add_function_url(auth_type=_lambda.FunctionUrlAuthType.AWS_IAM)

        # ---- Internal chat app: Cognito login -> CloudFront-hosted page -> HTTP API -> Lambda ----
        user_pool = cognito.UserPool(
            self, "Users",
            self_sign_up_enabled=False,  # admins invite users
            sign_in_aliases=cognito.SignInAliases(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=False)),
            password_policy=cognito.PasswordPolicy(min_length=12),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
        )
        user_pool_client = user_pool.add_client(
            "WebClient",
            generate_secret=False,  # browser app: no client secret
            auth_flows=cognito.AuthFlow(user_password=True),
            prevent_user_existence_errors=True,
        )

        site_bucket = s3.Bucket(
            self, "Site",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )
        distribution = cloudfront.Distribution(
            self, "SiteCdn",
            default_root_object="index.html",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(site_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
            ),
        )
        site_origin = f"https://{distribution.distribution_domain_name}"

        api = apigw.HttpApi(
            self, "Api",
            create_default_stage=False,
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=[site_origin],
                allow_methods=[apigw.CorsHttpMethod.POST],
                allow_headers=["Authorization", "Content-Type"],
            ),
        )
        api.add_routes(
            path="/ask",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("AskIntegration", fn),
            authorizer=HttpJwtAuthorizer(
                "CognitoAuthorizer",
                f"https://cognito-idp.{self.region}.amazonaws.com/{user_pool.user_pool_id}",
                jwt_audience=[user_pool_client.user_pool_client_id],
            ),
        )
        # Rate limit so one user or a leaked token cannot run up the Claude bill
        api_stage = apigw.HttpStage(
            self, "ApiStage", http_api=api, auto_deploy=True,
            throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10),
        )

        s3deploy.BucketDeployment(
            self, "DeploySite",
            destination_bucket=site_bucket,
            sources=[
                s3deploy.Source.asset(WEB_DIR),
                s3deploy.Source.json_data("config.json", {
                    "region": self.region,
                    "clientId": user_pool_client.user_pool_client_id,
                    "apiUrl": api_stage.url,
                }),
            ],
            distribution=distribution,
            distribution_paths=["/*"],
        )

        cdk.CfnOutput(self, "SiteUrl", value=site_origin)
        cdk.CfnOutput(self, "UserPoolId", value=user_pool.user_pool_id)
        cdk.CfnOutput(self, "DocsBucket", value=docs.bucket_name)
        cdk.CfnOutput(self, "KnowledgeBaseId", value=kb.attr_knowledge_base_id)
        cdk.CfnOutput(self, "DataSourceId", value=data_source.attr_data_source_id)
        cdk.CfnOutput(self, "QueryUrl", value=url.url)


app = cdk.App()
RagStack(app, "BedrockRagStack")
app.synth()

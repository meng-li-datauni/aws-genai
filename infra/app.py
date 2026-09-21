import aws_cdk as cdk
from aws_cdk import Stack, aws_bedrock as bedrock, aws_iam as iam, aws_lambda as _lambda
from aws_cdk import aws_s3 as s3, aws_s3vectors as s3v
from constructs import Construct

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

        cdk.CfnOutput(self, "DocsBucket", value=docs.bucket_name)
        cdk.CfnOutput(self, "KnowledgeBaseId", value=kb.attr_knowledge_base_id)
        cdk.CfnOutput(self, "DataSourceId", value=data_source.attr_data_source_id)
        cdk.CfnOutput(self, "QueryUrl", value=url.url)


app = cdk.App()
RagStack(app, "BedrockRagStack")
app.synth()

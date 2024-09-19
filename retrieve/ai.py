import sys

service_account_file_name = 'config/service_account_key.json'

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(service_account_file_name)

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])
  

import google.ai.generativelanguage as glm
generative_service_client = glm.GenerativeServiceClient(credentials=scoped_credentials)
retriever_service_client = glm.RetrieverServiceClient(credentials=scoped_credentials)
permission_service_client = glm.PermissionServiceClient(credentials=scoped_credentials)

corpus_resource_name = "corpora/community-health-ai-wtes3sbg1enp"

def query_ai(user_query):
    answer_style = "ABSTRACTIVE" # Or VERBOSE, EXTRACTIVE
    MODEL_NAME = "models/aqa"

    # Make the request
    # corpus_resource_name is a variable set in the "Create a corpus" section.
    content = glm.Content(parts=[glm.Part(text=user_query)])
    retriever_config = glm.SemanticRetrieverConfig(source=corpus_resource_name, query=content)
    req = glm.GenerateAnswerRequest(model=MODEL_NAME,
                                    contents=[content],
                                    semantic_retriever=retriever_config,
                                    answer_style=answer_style)
    aqa_response = generative_service_client.generate_answer(req)
    print(aqa_response)
        

    # Get the metadata from the first attributed passages for the source
    # chunk_resource_name = aqa_response.answer.grounding_attributions[0].source_id.semantic_retriever_chunk.chunk
    # get_chunk_response = retriever_service_client.get_chunk(name=chunk_resource_name)
    # print(get_chunk_response)

    chunk_name = aqa_response.answer.grounding_attributions[0].source_id.semantic_retriever_chunk.chunk
    document_resource_name = chunk_name[0:chunk_name[0:chunk_name.rfind('/')].rfind('/')]
    print(document_resource_name)
    get_document_request = glm.GetDocumentRequest(name=document_resource_name)
    document_response =  retriever_service_client.get_document(get_document_request)
    print(document_response)
    return [
        aqa_response.answer,
        document_response
    ]

if __name__ == "__main__":
    query = sys.argv[1]
    print(f"Querying for {query}")
    query_ai(query)

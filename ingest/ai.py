service_account_file_name = 'config/service_account_key.json'

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(service_account_file_name)

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])
  

import google.ai.generativelanguage as glm
generative_service_client = glm.GenerativeServiceClient(credentials=scoped_credentials)
retriever_service_client = glm.RetrieverServiceClient(credentials=scoped_credentials)
permission_service_client = glm.PermissionServiceClient(credentials=scoped_credentials)

def create_corpus(name):
    corpus = glm.Corpus(display_name=name)
    create_corpus_request = glm.CreateCorpusRequest(corpus=corpus)

    # Make the request
    create_corpus_response = retriever_service_client.create_corpus(create_corpus_request)

    # Set the `corpus_resource_name` for subsequent sections.
    corpus_resource_name = create_corpus_response.name
    print(create_corpus_response)


def get_corpus(name="corpora/community-health-ai-wtes3sbg1enp"):
    get_corpus_request = glm.GetCorpusRequest(name=name)

    # Make the request
    get_corpus_response = retriever_service_client.get_corpus(get_corpus_request)

    # Print the response
    print(get_corpus_response)

corpus_resource_name = "corpora/community-health-ai-wtes3sbg1enp"

def create_document(corpus, name, metadata = {}):
    # Create a document with a custom display name.
    document = glm.Document(display_name=name)

    # Add metadata.
    # Metadata also supports numeric values not specified here
    document_metadata = [
        glm.CustomMetadata(key=k, string_value=v) for k, v in metadata.items()
    ]
    document.custom_metadata.extend(document_metadata)

    # Make the request
    # corpus_resource_name is a variable set in the "Create a corpus" section.
    create_document_request = glm.CreateDocumentRequest(parent=corpus_resource_name, document=document)
    create_document_response = retriever_service_client.create_document(create_document_request)

    # Set the `document_resource_name` for subsequent sections.
    document_resource_name = create_document_response.name
    print(create_document_response)
    return create_document_response

def delete_document(name):
    delete_document_request = glm.DeleteDocumentRequest(name=name, force=True)
    delete_document_response = retriever_service_client.delete_document(delete_document_request)
    print(delete_document_response)


def create_document_and_add_chunks(name, metadata, chunks):
    document = create_document(corpus_resource_name, name, metadata)
    # chunk_1 = glm.Chunk(data={'string_value': "Chunks support user specified metadata."})
    # chunk_1.custom_metadata.append(glm.CustomMetadata(key="section",
    #                                                 string_value="Custom metadata filters"))
    chunks = [glm.Chunk(data={'string_value': chunk}) for chunk in chunks]
    create_chunk_requests = []
    for chunk in chunks:
        create_chunk_requests.append(glm.CreateChunkRequest(parent=document.name, chunk=chunk))

    # Make the request
    request = glm.BatchCreateChunksRequest(parent=document.name, requests=create_chunk_requests)
    response = retriever_service_client.batch_create_chunks(request)
    print(response)
    return document

def chunk_state(name):
    # Make the request
    request = glm.ListChunksRequest(parent=name)
    list_chunks_response = retriever_service_client.list_chunks(request)
    for index, chunks in enumerate(list_chunks_response.chunks):
        print(f'\nChunk # {index + 1}')
        print(f'Resource Name: {chunks.name}')
        # Only ACTIVE chunks can be queried.
        print(f'State: {glm.Chunk.State(chunks.state).name}')

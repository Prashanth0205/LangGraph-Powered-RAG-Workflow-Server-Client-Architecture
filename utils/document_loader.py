from typing import List
from document import Document # type: ignore
from langchain_community.document_loaders import FireCrawlLoader

class DocumentLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_docs(self, url: str) -> List[Document]:
        """
        Retrievers documents from the specified URL using the FireCrawlLoader
        """
        loader = FireCrawlLoader(
            api_key=self.api_key, url=url, mode='crawl',
        )
        raw_docs = loader.load()
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]

        return docs
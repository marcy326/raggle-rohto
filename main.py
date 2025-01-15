import json
import os
import sys
import time

from dotenv import load_dotenv
import requests
import pdfplumber

from langchain import callbacks
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Financial_Statements_2023.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Hada_Labo_Gokujun_Lotion_Overview.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Shibata_et_al_Research_Article.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/V_Rohto_Premium_Product_Information.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Well-Being_Report_2024.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """
    # 戻り値として質問に対する回答を返却してください。
    def download_and_load_pdfs(urls: list) -> list:
        """
        PDFファイルをダウンロードして読み込む関数

        Args:
            urls (list): PDFファイルのURLリスト

        Returns:
            documents (list): PDFファイルのテキストデータを含むDocumentオブジェクトのリスト

        Raises:
            Exception: ダウンロードまたは読み込みに失敗した場合に発生する例外

        Examples:
            >>> urls = ["https://example.com/example.pdf"]
            >>> download_and_load_pdfs(urls)
            [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
        """
        try:
            def download_pdf(url, save_path):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")
            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                with pdfplumber.open(tmp_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": url}
                        )
                    )
            return documents
        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    def create_vectorstore(docs: list) -> Chroma:
        """
        テキストデータからベクトルストアを生成する関数

        Args:
            docs (list): Documentオブジェクトのリスト

        Returns:
            vectorstore (Chroma): ベクトルストア

        Raises:
            Exception: ベクトルストアの生成に失敗した場合に発生する例外

        Examples:
            >>> docs = [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
            >>> create_vectorstore(docs)
            Chroma(...)
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            splitted_docs = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    metadata = doc.metadata.copy()
                    metadata['chunk_id'] = len(splitted_docs) + 1
                    splitted_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))

            embedding_function = OpenAIEmbeddings()

            vectorstore = Chroma.from_documents(
                splitted_docs,
                embedding_function,
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    docs = download_and_load_pdfs(pdf_file_urls)
    db = create_vectorstore(docs)
    retriever = db.as_retriever()

    template = """
    # ゴール
    あなたは、製薬企業に関する質問に対して、人間にわかりやすく、親切な回答を作成するアシスタントです。
    参考文章の部分には、回答を構成するために使用する提供された情報が含まれています。
    提供された参考文章は権威のあるものであり、それに疑問を抱いたり、内部知識を使用して修正しようとしないでください。
    回答は質問に対するものとして自然な形で作成してください。提供された情報に基づいていることを言及しないでください。
    一般的な回答は含めず、参考文章のみに基づき、質問に対してクリティカルな回答をしてください。
    提供された情報が空の場合は、答えがわからないと述べてください。

    # 質問
    {question}

    # 参考文章
    {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chat = ChatOpenAI(model=model)

    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | chat | output_parser

    answer = chain.invoke(question)

    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================

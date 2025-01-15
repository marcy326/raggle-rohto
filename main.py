import json
import os
import sys
import time

from dotenv import load_dotenv
import requests
import pdfplumber
from typing import List

from langchain import callbacks
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

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

    # ログレベルの設定を追加
    import logging
    logging.getLogger('chromadb').setLevel(logging.ERROR)

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
            # text_splitter = CharacterTextSplitter(
            #     separator="\n\n",
            #     chunk_size=512,
            #     chunk_overlap=64,
            #     length_function=len,
            # )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=512,
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
    
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10
        }
    )

    template = """
    あなたは製薬企業の専門アシスタントです。以下のガイドラインに従って質問に答えてください。

    # 回答ルール
    1. 正確性
    - 参考情報のみを使用し、推測や憶測はしない
    - 数値や事実は正確に伝える
    - 情報が不十分な場合は「十分な情報がありません」と回答

    2. 構成
    - 具体的な説明は省き、結論のみを回答
    - 箇条書きではなく文章形式で
    - 150字以内で簡潔に

    3. スタイル
    - 質問文を繰り返さない
    - 受動態ではなく能動態を使用

    4. コンテキスト管理
    - 参考情報にない質問には回答しない
    - 回答に不要な参考情報の内容は無視する

    # 質問
    {question}

    # 参考情報
    {context}

    # 回答例
    以下はあくまで回答例であるため、参考情報に従って回答してください。

    - 質問: 存在意義（パーパス）は、なんですか？
    - 回答: 世界の人々に商品やサービスを通じて「健康」をお届けすることによって、当社を取り巻くすべての人や社会を「Well-being」へと導き、明日の世界を元気にすることです。

    - 質問: 事務連絡者の電話番号は？
    - 回答: （06）6758-1235です。
    
    - 質問: Vロートプレミアムは、第何類の医薬品ですか？
    - 回答: 第2類医薬品です。

    - 質問: 肌ラボ 極潤ヒアルロン液の詰め替え用には、何mLが入っていますか？
    - 回答: 170mLが入っています。

    - 質問: LN211E8は、どのようなhiPSCの分化において、どのように作用しますか？
    - 回答: Wnt 活性化を通じて神経堤細胞への分化を促進します。

    """

    prompt = ChatPromptTemplate.from_template(template)

    chat = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1 
    )

    # HyDE用のプロンプトテンプレート
    hyde_template = """
    あなたは専門的な情報検索システムの一部です。以下の質問に対して、以下のガイドラインに従って仮想的な回答を生成してください。

    【ガイドライン】
    1. 回答形式
    - 質問の意図を正確に反映
    - 専門用語を適切に使用
    - 具体的な数値や事実を含む
    - 150字以内で簡潔に

    2. 内容
    - 実際のデータに基づくのではなく、質問から推測される形式の回答
    - 専門家が書いたような文体で
    - 関連するキーワードを自然に含む

    3. スタイル
    - 「です・ます」調を使用
    - 箇条書きではなく文章形式
    - 受動態ではなく能動態

    【例】
    質問: 本社の所在地は？
    回答: ロート製薬の本社は大阪府大阪市中央区道修町1丁目8番1号に所在しています。最寄り駅は地下鉄堺筋線の北浜駅で、徒歩約5分の場所に位置しています。

    質問: 代表電話番号は？
    回答: ロート製薬の代表電話番号は06-6758-1235です。受付時間は平日の9時から17時までとなっています。

    【現在の質問】
    {question}

    【仮回答】
    """
    
    # HyDEの実装
    def generate_hypothetical_answer(question: str) -> str:
        hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
        hyde_llm = ChatOpenAI(
            temperature=0.7,
            model=model,
            max_tokens=200,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.2
        )
        hyde_chain = hyde_prompt | hyde_llm | StrOutputParser()
        return hyde_chain.invoke({"question": question})

    hypothetical_answer = generate_hypothetical_answer(question)

    # クエリ拡張の実装
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""
        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        あなたはAI言語モデルアシスタントです。
        以下のユーザーの質問に対して、ベクトルデータベースから関連文書を取得するための5つの異なるバージョンの質問を生成してください。
        複数の視点から質問を生成することで、距離ベースの類似性検索の制限を克服するのに役立ちます。
        生成された質問は改行で区切ってください。
        
        元の質問: {question}
        
        生成する質問は日本語で、以下の点に注意してください。
        1. 専門用語の言い換えを含める
        2. 関連する概念を含める
        3. 具体的な事例を含める
        4. 異なる視点からの質問を含める
        5. 簡潔で明確な表現を使用する
        """,
    )

    llm = ChatOpenAI(temperature=0.3, model=model)
    llm_chain = QUERY_PROMPT | llm | LineListOutputParser()

    # クエリ拡張を既存のパイプラインに統合
    expanded_queries = llm_chain.invoke({"question": question})
    all_queries = [question, hypothetical_answer] + expanded_queries

    # 複数クエリでドキュメントを取得
    retrieved_docs = []
    for query in all_queries:
        retrieved_docs.extend(retriever.invoke(query))
    
    # 重複ドキュメントの削除
    unique_docs = []
    seen = set()
    for doc in retrieved_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    

    # 既存のパイプラインを更新
    n_doc = -1
    setup_and_retrieval = RunnableParallel(
        {
            "context": lambda _: "\n\n".join([d.page_content for d in unique_docs][:n_doc]),
            "question": RunnablePassthrough()
        }
    )

    output_parser = StrOutputParser()

    chain = setup_and_retrieval | prompt | chat | output_parser

    answer = chain.invoke(question)

    # デバッグ情報収集用
    def calculate_tokens(text: str) -> int:
        # 簡易的なトークン計算（1トークン≈4文字）
        return len(text) // 4

    context = "\n\n".join([
        f"【ドキュメント {i+1}】\n{doc.page_content}" 
        for i, doc in enumerate(unique_docs[:n_doc])  # 最大2ドキュメント
    ])

    # コンテキスト情報のデバッグ出力
    debug_info = {
        "total_docs": len(unique_docs),
        "selected_docs": min(n_doc, len(unique_docs)),
        "context_length": len(context),
        "context_tokens": calculate_tokens(context),
        "doc_lengths": [len(d.page_content) for d in unique_docs[:n_doc]],
        "doc_tokens": [calculate_tokens(d.page_content) for d in unique_docs[:n_doc]],
        "question_length": len(question),
        "question_tokens": calculate_tokens(question)
    }
    # print(debug_info)

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

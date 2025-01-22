import json
import os
import sys
import time

from dotenv import load_dotenv
import requests
import pdfplumber
from typing import List, Any
from pydantic import BaseModel, Field

from langchain import callbacks
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

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

    # ============================================================
    # データ処理関数群
    # ============================================================

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

        def extract_tables_as_markdown(page) -> tuple[str, list]:
            """
            PDFページからテーブルを抽出し、Markdown形式に変換する関数

            Args:
                page: pdfplumberのページオブジェクト

            Returns:
                tuple: (Markdown形式のテーブル文字列, テーブル領域のリスト)
            """
            markdown_tables = ""
            table_areas = []
            tables = page.find_tables()
            
            for table in tables:
                if table:
                    # テーブル領域を記録
                    bbox = (
                        max(0, table.bbox[0]),  # x0
                        max(0, table.bbox[1]),  # y0
                        min(page.width, table.bbox[2]),  # x1
                        min(page.height, table.bbox[3])  # y1
                    )
                    table_areas.append(bbox)
                    
                    # テーブルデータを抽出
                    table_data = table.extract()
                    
                    if table_data:
                        # セルデータを整理
                        sanitized_table = [[sanitize_cell(cell) for cell in row] for row in table_data]
                        
                        # Markdownテーブル作成
                        md_table = "| " + " | ".join(sanitized_table[0]) + " |\n"
                        md_table += "| " + " | ".join(["---"] * len(sanitized_table[0])) + " |\n"
                        for row in sanitized_table[1:]:
                            md_table += "| " + " | ".join(row) + " |\n"
                        markdown_tables += "\n" + md_table + "\n"
            
            return markdown_tables, table_areas

        def sanitize_cell(cell) -> str:
            """
            セルの内容を整理する

            Args:
                cell: テーブルセルの内容

            Returns:
                str: 整理されたセル内容
            """
            if cell is None:
                return ""
            return ' '.join(str(cell).split())

        def extract_text_excluding_tables(page, table_areas) -> str:
            """
            テーブル領域を除外してテキストを抽出する関数

            Args:
                page: pdfplumberのページオブジェクト
                table_areas: 除外するテーブル領域のリスト

            Returns:
                str: テーブル領域を除外したテキスト
            """
            if not table_areas:
                return page.extract_text() or ""
            
            # テーブル領域をマスク
            cropped_page = page
            for area in table_areas:
                try:
                    cropped_page = cropped_page.outside_bbox(area)
                except ValueError:
                    # 境界ボックスがページ外に出ている場合、スキップ
                    continue
            
            return cropped_page.extract_text() or ""
        
        def download_pdf(url, save_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download {url}")

        try:
            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                with pdfplumber.open(tmp_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        # テーブルを先に抽出
                        markdown_tables, table_areas = extract_tables_as_markdown(page)
                        
                        # テーブル領域を除外してテキストを抽出
                        text = extract_text_excluding_tables(page, table_areas)
                        if text:
                            full_text += text + "\n"
                        
                        # テーブルを追加
                        full_text += markdown_tables

                    # 元のPDFテキストをファイルに保存
                    with open(f"original_pdf_{i}.txt", "w", encoding="utf-8") as f:
                        f.write(full_text)

                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": url}
                        )
                    )
            return documents
        
        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    def setup_retriever(docs: list) -> Chroma:
        """ParentDocumentRetrieverを使用したベクトル検索を設定"""
        try:
            # 2種類のテキストスプリッターを用意
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=512,
                length_function=len,
            )
            
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=128,
                chunk_overlap=32,
                length_function=len,
            )

            # ストレージの初期化
            vectorstore = Chroma(
                embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
            )
            docstore = InMemoryStore()

            # ParentDocumentRetrieverの初期化
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter
            )
            
            # ドキュメントを追加
            retriever.add_documents(docs)
            
            return retriever

        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")
    
    def generate_hypothetical_answer(question: str) -> str:
        """HyDEを用いて仮想回答を生成"""
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

    def expand_queries(question: str) -> List[str]:
        """クエリ拡張を行い、複数の関連クエリを生成"""
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
        return llm_chain.invoke({"question": question})
    
    def rerank_documents(docs: List[Document], question: str) -> List[Document]:
        """ドキュメントを再ランク付け"""
        try:
            class RankedIndices(BaseModel):
                ranked_indices: List[int] = Field(
                    ...,
                    description="ランク付けされたドキュメントのインデックスリスト",
                    example=[1, 4, 0, 3, 2]
                )
            
            output_parser = PydanticOutputParser(pydantic_object=RankedIndices)
            format_instructions = output_parser.get_format_instructions()

            rerank_template = """
            以下のドキュメントを、質問との関連性に基づいてランク付けしてください。
            最も関連性の高いドキュメントを最初に、関連性の低いドキュメントを後に配置します。

            質問: {question}

            ドキュメント:
            {documents}

            ランク付けの基準:
            1. 質問のキーワードとの一致度
            2. 文脈的な関連性
            3. 情報の具体性
            4. 信頼性の高さ

            出力形式:
            {format_instructions}
            """
            
            # ドキュメントを簡潔にまとめる
            docs_concat = "\n\n".join([
                f"【ドキュメント {i}】\n{d.page_content}..." 
                for i, d in enumerate(docs)
            ])
            
            prompt = ChatPromptTemplate.from_template(rerank_template)
            llm = ChatOpenAI(temperature=0, model=model)
            chain = prompt | llm | output_parser
            
            # ランク付け結果を取得
            result = chain.invoke({
                "question": question,
                "documents": docs_concat,
                "format_instructions": format_instructions
            })
            
            # 結果をパースしてドキュメントを並べ替え
            ranked_indices = [docs[i] for i in result.ranked_indices]
            return ranked_indices
        
        except Exception as e:
            return docs  # 失敗時は元の順序を保持

    def retrieve_documents(retriever: Any, queries: List[str]) -> List[Document]:
        """複数のクエリを使用して関連ドキュメントを取得"""
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(retriever.invoke(query))
        return retrieved_docs

    def remove_duplicate_docs(docs: List[Document]) -> List[Document]:
        """重複ドキュメントを削除"""
        unique_docs = []
        seen = set()
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs
    
    def setup_rag_chain(unique_docs: List[Document]) -> Any:
        """RAGパイプラインを構築"""
        template = """
        あなたは製薬企業の専門アシスタントです。以下のガイドラインに従って質問に答えてください。

        # 回答ルール
        1. 正確性
        - 参考情報に基づいた事実のみを回答
        - 推測や憶測は一切しない
        - 数値や事実は正確に伝える
        - 情報が不十分な場合は「十分な情報がありません」と回答

        2. 有用性
        - 質問の意図を正確に理解して回答
        - 必要な補足情報を適切に追加
        - 関連する重要な情報を過不足なく含める
        - ユーザーが求める情報を的確に提供

        3. 簡潔性
        - 具体的な説明はしない
        - 結論のみを簡潔に述べる
        - 100字以内で簡潔に
        - 質問文を繰り返さない

        5. コンテキスト管理
        - 参考情報にない質問には回答しない
        - 回答に不要な情報は含めない

        6. 安全性
        - 医療アドバイスや診断は一切行わない
        - 製品の誤用を招く可能性のある情報は提供しない
        - 個人の健康状態に関する具体的なアドバイスはしない
        - 法的・倫理的に問題のある内容は回答しない

        7. 倫理的配慮
        - 差別的・偏見的な表現は一切使用しない
        - 個人情報に関連する質問には回答しない
        - 製品の安全性に関する誤解を招く表現は避ける
        - 医療行為を推奨するような表現はしない

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

        - 質問: この薬を子供に使っても大丈夫ですか？
        - 回答: 製品の使用に関しては、必ず医師や薬剤師にご相談ください。年齢による適応や使用方法は製品によって異なります。

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
        
        output_parser = StrOutputParser()

        setup_and_retrieval = RunnableParallel(
            {
                "context": lambda _: "\n\n".join([
                    f"【文脈 {i+1}/{len(unique_docs)}】\n{d.page_content}"
                    for i, d in enumerate(unique_docs)
                ]),
                "question": RunnablePassthrough()
            }
        )

        return setup_and_retrieval | prompt | chat | output_parser

    # ============================================================
    # 主要処理フロー
    # ============================================================

    # PDFデータの読み込み
    docs = download_and_load_pdfs(pdf_file_urls)
    
    # 検索器の設定
    retriever = setup_retriever(docs)

    # HyDEによる仮想回答生成
    hypothetical_answer = generate_hypothetical_answer(question)

    # クエリ拡張
    expanded_queries = expand_queries(question)
    all_queries = [question, hypothetical_answer] + expanded_queries

    # ドキュメント取得
    retrieved_docs = retrieve_documents(retriever, all_queries)
    
    # 重複ドキュメントの削除
    unique_docs = remove_duplicate_docs(retrieved_docs)

    # Reranking
    reranked_docs = rerank_documents(unique_docs, question)

    # RAGチェーンの構築
    chain = setup_rag_chain(reranked_docs)

    # RAGチェーンの実行
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

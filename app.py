from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def get_llm_response(user_input, specialist_type):
    """
    LLMからの回答を取得する関数
    
    Args:
        user_input (str): ユーザーからの入力テキスト
        specialist_type (str): 選択された専門家の種類
    
    Returns:
        str: LLMからの回答
    """
    # LLMの初期化
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # 専門家の種類に応じてシステムメッセージを設定
    specialist_prompts = {
        "外科医": "あなたは経験豊富な外科医です。医学的な知識を活用して、患者の質問に対して適切なアドバイスを提供してください。",
        "内科医": "あなたは経験豊富な内科医です。内科全般の医学的知識を活用して、患者の質問に対して適切なアドバイスを提供してください。",
        "小児科医": "あなたは経験豊富な小児科医です。小児医療の専門知識を活用して、子どもの健康に関する質問に適切なアドバイスを提供してください。",
        "整形外科医": "あなたは経験豊富な整形外科医です。骨、関節、筋肉に関する専門知識を活用して、患者の質問に対して適切なアドバイスを提供してください。"
    }
    
    # メッセージの構成
    messages = [
        SystemMessage(content=specialist_prompts[specialist_type]),
        HumanMessage(content=user_input),
    ]
    
    # LLMに送信して回答を取得
    result = llm(messages)
    return result.content

# Streamlitアプリケーションのメイン部分
def main():
    st.title("🏥 AI医療相談アプリ")
    
    # アプリケーションの概要と操作方法
    st.markdown("""
    ## 📋 アプリケーション概要
    このアプリケーションは、LangChainとOpenAI GPTを活用した医療相談システムです。
    4つの専門分野の医師（外科医、内科医、小児科医、整形外科医）のいずれかを選択して、
    医療に関する質問をすることができます。
    
    ## 📖 操作方法
    1. **専門家を選択**: ラジオボタンから相談したい医師の専門分野を選択してください
    2. **質問を入力**: テキストエリアに医療に関する質問や相談内容を入力してください
    3. **回答を取得**: 「回答を取得」ボタンをクリックして、AI医師からの回答を確認してください
    
    ⚠️ **注意**: このアプリケーションはAIによる情報提供のみを目的としており、実際の医療診断や治療の代替ではありません。
    重要な健康問題については、必ず実際の医療機関を受診してください。
    """)
    
    st.markdown("---")
    
    # 専門家選択用のラジオボタン
    st.subheader("👨‍⚕️ 相談する専門家を選択してください")
    specialist_type = st.radio(
        "専門分野:",
        ("外科医", "内科医", "小児科医", "整形外科医"),
        index=0
    )
    
    # 入力フォーム
    st.subheader("💬 ご質問・ご相談内容")
    user_input = st.text_area(
        "質問を入力してください:",
        placeholder="例: 膝の痛みが続いているのですが、どのような原因が考えられますか？",
        height=150
    )
    
    # 回答取得ボタン
    if st.button("🔍 回答を取得", type="primary"):
        if user_input:
            with st.spinner(f"{specialist_type}が回答を準備しています..."):
                try:
                    # LLMから回答を取得
                    response = get_llm_response(user_input, specialist_type)
                    
                    # 回答の表示
                    st.subheader(f"📝 {specialist_type}からの回答")
                    st.success(response)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    st.info("OpenAI APIキーが正しく設定されているかご確認ください。")
        else:
            st.warning("質問を入力してください。")
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Powered by LangChain & OpenAI GPT-4o-mini
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
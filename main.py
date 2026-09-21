from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# MEMORY STORE
# =========================================================

store = {}


def get_session_history(session_id):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# =========================================================
# MODEL
# =========================================================

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",
            """
You are AgentLens AI.

You are a helpful AI assistant.

Your job is to understand the user's question
and provide a clear, accurate and useful answer.

Rules:

1. Answer the user's question clearly.
2. Stay relevant to the question.
3. Do not invent information.
4. If you are uncertain, clearly say so.
5. Follow the user's instructions.
6. Use previous conversation when relevant.
7. Keep the response easy to understand.
"""
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{question}"
        )

    ]
)


# =========================================================
# CHATBOT
# =========================================================

chatbot_chain = prompt | model


# =========================================================
# ADD MESSAGE HISTORY
# =========================================================

chatbot = RunnableWithMessageHistory(chatbot_chain,get_session_history,input_messages_key="question",
                                     history_messages_key="history"
)


# =========================================================
# CHAT FUNCTION
# =========================================================

def chat(message, session_id="user_1"):

    response = chatbot.invoke(

        {
            "question": message
        },

        config={"configurable": {"session_id": session_id}}

    )

    return response.content


# =========================================================
# USER SESSION
# =========================================================

session_id = input(
    "Enter User ID: "
)


# =========================================================
# CHAT LOOP
# =========================================================

print("\n🤖 AgentLens AI")
print("Type 'bye' to exit.\n")


while True:

    question = input("You: ")

    if question.lower() == "bye":

        print("🤖: Thank you! Have a nice day.")

        break


    answer = chat(question,session_id)

    print("🤖:",answer)


# =========================================================
# DISPLAY USER MEMORY
# =========================================================

print("\n===== Conversation Memory =====\n")


for message in get_session_history(session_id).messages:

    print(message.type,":",message.content)
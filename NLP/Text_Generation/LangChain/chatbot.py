import math
import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class LangChainChatbot:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create tools
        def safe_calculator(expr: str) -> str:
            """Safely evaluate simple math expressions without eval vulnerabilities."""
            allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed_names.update({"abs": abs, "round": round})
            try:
                code = compile(expr, "<calc>", "eval")
                for name in code.co_names:
                    if name not in allowed_names:
                        return "Disallowed identifier in expression"
                result = eval(code, {"__builtins__": {}}, allowed_names)
                return str(result)
            except Exception as e:
                return f"Calculation error: {e}"

        self.tools = [
            Tool(
                name="Calculator",
                func=safe_calculator,
                description="Safe calculator for basic math and common functions (sin, cos, log, etc.)",
            ),
            Tool(
                name="Search",
                func=lambda x: f"Searching for: {x}",
                description="Useful for searching information",
            ),
        ]

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
        )

        # Create conversation chain
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)

    def chat(self, user_input):
        """Process user input and return response"""
        try:
            # Use agent for tool-based interactions
            if any(keyword in user_input.lower() for keyword in ["calculate", "search"]):
                response = self.agent.run(user_input)
            else:
                # Use conversation chain for general chat
                response = self.conversation.predict(input=user_input)

            return response
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    # Initialize chatbot
    chatbot = LangChainChatbot()

    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        response = chatbot.chat(user_input)
        print(f"\nBot: {response}")


if __name__ == "__main__":
    main()

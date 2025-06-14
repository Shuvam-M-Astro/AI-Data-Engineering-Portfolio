from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LangChainChatbot:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="Calculator",
                func=lambda x: eval(x),
                description="Useful for performing calculations"
            ),
            Tool(
                name="Search",
                func=lambda x: f"Searching for: {x}",
                description="Useful for searching information"
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        # Create conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
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
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.chat(user_input)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    main() 
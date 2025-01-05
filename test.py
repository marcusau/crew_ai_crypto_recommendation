import os
import re
from datetime import datetime
import pandas as pd
import requests
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from crewai import Agent, LLM
#from google.colab import userdata
#from langchain.agents import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import Tool
from typing import Union, Any, Optional
import uuid
from langchain.callbacks.manager import Callbacks
from langchain.schema.runnable import RunnableConfig

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "https://api.your-provider.com/v1"
os.environ["OPENAI_MODEL_NAME"] = "your-model-name"
#os.environ["OPENAI_API_KEY"] = "NA"

llm = LLM(
    model="gpt-4o",
    temperature=0.8,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42
)

llm = LLM(
    provider="openai",
    model="gpt-4",  # Use "gpt-4" for GPT-4
    api_key=os.environ["OPENAI_API_KEY"]
)

response = llm.generate_text(
    prompt="Tell me a joke about AI.",
    max_tokens=50
)

print(response)
#llm=LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

# ticker = "BTC-USD"
# def get_daily_closing_prices(ticker:str) -> pd.DataFrame:
#         symbol = yf.Ticker(ticker)
#         df = symbol.history(period="3mo")
#         df1 = df[["Close"]]
#         df1.rename(columns={"Close":"price"}, inplace = True)
#         df1.index = pd.to_datetime(df.index)
#         return df1
    
# def crypto_price_tool(ticker_symbol: str) -> str:
#         """Get daily closing price for a given ticker symbol"""
#         price_df = get_daily_closing_prices(ticker_symbol)
#         text_output = ["date - price (USD)"]
#         for date, row in price_df.iterrows():
#             text_output.append(f"{date.strftime('%Y-%m-%d')} - {row['price']:.2f}")
#         return "\n".join(text_output)
# print(crypto_price_tool("BTC-USD"))
# # ########################################################################## Load human tools to enhance the AI's capabilities 


def analyze_crypto(crypto_ticker: str):
    customer_communicator_agent = Agent(
        role="Experienced customer communicator",
        goal=f"Analyze {crypto_ticker} cryptocurrency",
        backstory="""You're highly experienced in communicating about crypto with customers and their research needs""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=5,
        memory=True,
        #tools=[human_tool],
    )

    get_crypto_task = Task(
        description=f"Analyze {crypto_ticker} cryptocurrency",
        expected_output="""crypto tickers that the human wants you to research - These are the crypto tickers you know = for example -
        {
        'BTC-USD':'Bitcoin USD',
    'ETH-USD':'Ethereum USD',
    'USDT-USD':'Tether USDt USD',
    'XRP-USD':'XRP USD',
    'BNB-USD':'BNB USD',
    'SOL-USD':'Solana USD',
    'DOGE-USD':'Dogecoin USD',
    'USDC-USD':'USD Coin USD',
    'STETH-USD':'Lido Staked ETH USD',
    'ADA-USD':'Cardano USD',
    'WTRX-USD':'Wrapped TRON USD',
    'TRX-USD':'TRON USD',
    'WSTETH-USD':'Lido wstETH USD',
    'AVAX-USD':'Avalanche USD',
    'TON11419-USD':'Toncoin USD',
    'LINK-USD':'Chainlink USD',
    'SHIB-USD':'Shiba Inu USD',
    'WBTC-USD':'Wrapped Bitcoin USD',
    'SUI20947-USD':'Sui USD',
    'WETH-USD':'WETH USD',
    'HBAR-USD':'Hedera USD',
    'DOT-USD':'Polkadot USD',
    'XLM-USD':'Stellar USD',
    'H HYPE32196-USD':'Hyperliquid USD',
    'BCH-USD':'Bitcoin Cash USD',
    }.""",
        agent=customer_communicator_agent,
    )
    # ###########################################################################

    def get_daily_closing_prices(ticker:str) -> pd.DataFrame:
        symbol = yf.Ticker(ticker)
        df = symbol.history(period="3mo")
        df1 = df[["Close"]]
        df1.rename(columns={"Close":"price"}, inplace = True)
        df1.index = pd.to_datetime(df.index)
        return df1
    # price_df = get_daily_closing_prices("BTC-USD")
    # print(price_df.tail(5))

    @tool("price tool")
    def crypto_price_tool(ticker_symbol: str) -> str:
        """Get daily closing price for a given ticker symbol"""
        price_df = get_daily_closing_prices(ticker_symbol)
        text_output = ["date - price (USD)"]
        for date, row in price_df.iterrows():
            text_output.append(f"{date.strftime('%Y-%m-%d')} - {row['price']:.2f}")
        return "\n".join(text_output)
#     #print(crypto_price_tool("BTC-USD"))

    price_analyst_agent = Agent(
        role="Crypto Price Analyst",
        goal=f"""Get historical prices for {crypto_ticker}. Write 1 paragraph analysis of the market and make prediction - up, down or neutral.""",
        backstory="""You're an expert analyst of trends based on crypto historical prices. You have a complete understanding of macroeconomic factors, but you specialize into technical analys based on historical prices.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[crypto_price_tool])

    get_price_analysis_task = Task(
        description=f"""
        Use the crypto_price_tool to get historical prices for {crypto_ticker}

        The current date is {datetime.today()}.

        Compose the results into a helpful report""",

        expected_output="""Create 1 paragraph summary for the crypto, along with a prediction for the future trend""",
        agent=price_analyst_agent,
        context=[get_crypto_task],
    )
#     # # # ###################################################################################

    # writer_agent = Agent(
    #     role="Report Writer",
    #     goal=f"""Write 1 paragraph report of {crypto_ticker} market.""",
    #     backstory="""
    #     You're highly respected as an exceptional market analyst with extensive experience tracking crypto assets consistently for more than a decade. 
    #     Your insights and projections are notably precise, establishing a strong reputation in the crypto sphere.
    #     Alongside your deep knowledge of traditional crypto, you have a nuanced understanding of human behavior and major economic forces.
    #     You seamlessly integrate different frameworks, like cyclical theories, and take a multifaceted approach to each analysis, 
    #     adeptly balancing various perspectives.
    #     While you monitor news and price history, you view them with a critical lens, carefully evaluating source reliability. 
    #     Your standout skill is your ability to translate complex market insights into straightforward summaries, 
    #     making intricate concepts approachable for all audiences.
    #     Your approach to writing includes:
    #     Bullet-pointed executive summaries that emphasize the key takeaways
    #     Streamlined explanations that distill complex insights into core ideas
    #     You excel at turning highly technical content into compelling, accessible narratives, making even the most 
    #     challenging topics clear and engaging for readers""",
    #     verbose=True,
    #     allow_delegation=False,
    #     llm=llm,
    #     max_iter=5,
    #     memory=True,
    #    )

    # write_report_task = Task(
    #     description=f"""Use the reports from the news analyst and the price analyst to create a report that summarizes {crypto_ticker}""",

    #     expected_output=f"""Perform a detailed technical analysis of {crypto_ticker} using the daily closing price data for 3 months,
    #     . Focus on identifying key price patterns, support and resistance levels, and trend directions.
    #     Utilize indicators such as moving averages (SMA, EMA), Bollinger Bands, and RSI to gauge momentum and volatility.
    #     Examine volume trends to confirm price movements and assess the impact of dividends and stock splits on the price action.
    #     Additionally, incorporate candlestick patterns to predict potential reversals or continuations and provide insights into the stock's future price movements
    #     and make prediction - up, down or neutral. Do not make up false information & give rationale of every point that is analyzed.""",
    #     agent=writer_agent,
    #     context=[get_price_analysis_task],
    # )

    crew = Crew(
        agents=[customer_communicator_agent,price_analyst_agent],#,writer_agent],
        tasks=[get_crypto_task,get_price_analysis_task],#, write_report_task],
        verbose=True,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
        manager_llm=llm,
        max_iter=15,
    )

    results = crew.kickoff()
    return results

# Example usage:
if __name__ == "__main__":
    # You can input any crypto ticker from the list above
    crypto_ticker = "BTC-USD"  # Example ticker
    analysis_results = analyze_crypto(crypto_ticker)
    print(analysis_results)
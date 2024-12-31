import os
import re
from datetime import datetime
import pandas as pd
import requests
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
#from google.colab import userdata
from langchain.agents import load_tools
#from langchain_community.agent_toolkits.load_tools import load_tools
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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model_name= "gpt-4o"  #gpt-4o'   #"gpt-4o",", "gpt-4o-mini"
print(f"Model name: {model_name}")
llm = ChatOpenAI(
    model=model_name,   #"gpt-4o",
    temperature=0,
    seed=42,
    max_retries=2
)

class HumanInputInput(BaseModel):
    tool_input: Union[str, dict[str, Any]]
    verbose: Optional[bool] = None
    start_color: Optional[str] = None
    color: Optional[str] = None
    callbacks: Optional[Callbacks] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    run_name: Optional[str] = None
    run_id: Optional[uuid.UUID] = None
    config: Optional[RunnableConfig] = None
    tool_call_id: Optional[str] = None
    kwargs: Any = None

########################################################################## Load human tools to enhance the AI's capabilities 
human_tools = load_tools(["human"])

# Create a proper Tool instance for human input
human_tool = Tool(
    name="Human Input",
    description="Tool for getting input from a human",
    func=human_tools[0].run
)

customer_communicator = Agent(
    role="Experienced customer communicator",
    goal="Find which crytpo ticker the customer is interested in",
    backstory="""You're highly experienced in communicating about crypto with customers and their research needs""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[human_tool],
)

get_equity = Task(
    description=f"Ask which crypto ticker the customer is interested in.",
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
    agent=customer_communicator,
)
###########################################################################

def get_daily_closing_prices(ticker:str) -> pd.DataFrame:
    msft = yf.Ticker(ticker)
    df = msft.history(period="1mo")
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
    text_output = []
    for date, row in price_df.iterrows():
        text_output.append(f"{date.strftime('%Y-%m-%d')} - {row['price']:.2f}")
    return "\n".join(text_output)


price_analyst = Agent(
    role="Crypto Price Analyst",
    goal="""Get historical prices for a given crypto ticker. Write 1 paragraph analysis of the market and make prediction - up, down or neutral.""",
    backstory="""You're an expert analyst of trends based on crypto historical prices. You have a complete understanding of macroeconomic factors, but you specialize into technical analys based on historical prices.  """,
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[crypto_price_tool])

get_price_analysis = Task(
    description=f"""
    Use the price tool to get historical prices

    The current date is {datetime.now()}.

    Compose the results into a helpful report""",

    expected_output="""Create 1 paragraph summary for the stock, along with a prediction for the future trend  """,
    agent=price_analyst,
    context=[get_equity],
)
###################################################################################

writer = Agent(
    role="Report Writer",
    goal="""Write 1 paragraph report of the stock market.""",
    backstory="""
    You're highly respected as an exceptional market analyst with extensive experience tracking assets consistently for more than a decade. 
    Your insights and projections are notably precise, establishing a strong reputation in the crypto sphere.
    Alongside your deep knowledge of traditional crypto, you have a nuanced understanding of human behavior and major economic forces.
    You seamlessly integrate different frameworks, like cyclical theories, and take a multifaceted approach to each analysis, 
    adeptly balancing various perspectives.
    While you monitor news and price history, you view them with a critical lens, carefully evaluating source reliability. 
    Your standout skill is your ability to translate complex market insights into straightforward summaries, 
    making intricate concepts approachable for all audiences.
    Your approach to writing includes:
    Bullet-pointed executive summaries that emphasize the key takeaways
    Streamlined explanations that distill complex insights into core ideas
    You excel at turning highly technical content into compelling, accessible narratives, making even the most 
    challenging topics clear and engaging for readers""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=5,
    memory=True,
   )

write_report = Task(
    description=f"""Use the reports from the news analyst and the price analyst to create a report that summarizes the stock""",

    expected_output="""Perform a detailed technical analysis of a crypto using the daily closing price data for 3 months,
    . Focus on identifying key price patterns, support and resistance levels, and trend directions.
    Utilize indicators such as moving averages (SMA, EMA), Bollinger Bands, and RSI to gauge momentum and volatility.
    Examine volume trends to confirm price movements and assess the impact of dividends and stock splits on the price action.
    Additionally, incorporate candlestick patterns to predict potential reversals or continuations and provide insights into the stock's future price movements
    and make prediction - up, down or neutral. Do not make up false information & give rationale of every point that is analyzed.""",
    agent=writer,
    context=[ get_price_analysis],
)
# # Using requests and BeautifulSoup instead of requests-html
# import requests
# from bs4 import BeautifulSoup

# num_currencies = 40
# url = f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}"
# headers = {'User-Agent': 'Mozilla/5.0'}
# response = requests.get(url, headers=headers)
# soup = BeautifulSoup(response.content, 'html.parser')

# # Parse the table using pandas
# tables = pd.read_html(response.text)
# df = tables[0].copy()
# df.rename(columns={"Symbol":"ticker"}, inplace = True)
# df=df[["ticker","Name"]]
# df['Name']=df['Name'].replace("USD","").str.strip()
# df['quote']="'"+ df['ticker']+"':'"+df['Name']+"',"
# for quote in df['quote'].tolist():
#     print(quote)
###########################################################################################
crew = Crew(
    agents=[customer_communicator, price_analyst, writer],
    tasks=[get_equity, get_price_analysis, write_report],
    verbose=True,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)

results = crew.kickoff()
print(results)
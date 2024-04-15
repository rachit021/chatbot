import os
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql.types import *
from pyspark.sql.types import StructType, StringType, LongType, DoubleType, StructField
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from langchain_experimental.agents.agent_toolkits.spark.base import create_spark_dataframe_agent
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool
import certifi
import ssl

# Update the SSL context to use the system's root certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Read the CSV file
SalesDF = spark.read.csv('/mnt/adls/chile/dev/machine_learning/test_llm1.csv/', header=True)

# Initialize OpenAI model
chatgpt = OpenAI(model_name="gpt-3.5-turbo-1106")

# Create the agent with a customized prompt and the PythonREPLTool
agent_executor = create_spark_dataframe_agent(
    chatgpt, SalesDF, additional_tools=[PythonREPLTool()], prompt="""
You are an intelligent agent tasked with analyzing a sales data DataFrame and answering questions related to promotions, volumes, and other sales metrics. The DataFrame contains columns like 'promo id', 'planned baseline volume units', 'actual baseline volume', and others. Use the provided tools and your knowledge to accurately interpret the data and provide helpful responses to the questions asked. You have access to the PySpark library and can use its functions to perform operations on the DataFrame, such as filtering, aggregating, sorting, and more. If you need to perform any data manipulation or calculations, feel free to use the Python REPL tool and leverage PySpark functions.""")

# Define the chatbot function
def chatbot(input_text):
    if input_text:
        reply = agent_executor.run(input_text)
        return reply

# Export the chatbot function
export chatbot

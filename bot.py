from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from textblob import TextBlob
import pandas as pd
import numpy as np
import seaborn as sns
import pyspark
from pyspark.sql.types import *
from pyspark.sql.types import StructType, StringType, LongType, DoubleType, StructField
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

import os
from langchain_experimental.agents.agent_toolkits.spark.base import create_spark_dataframe_agent
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool
import gradio as gr

app = FastAPI(title="SaamaBot")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.get("/SaamaBot/{text}")

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

#SalesDF=spark.read.csv(r'C:\Users\rachit.choudhary\Downloads\test_llm1.csv')

schema = StructType([
    StructField("Promo ID", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Brand", StringType(), True),
    StructField("Product", StringType(), True),
    StructField("Promo Name", StringType(), True),
    StructField("Planned Baseline Volume (Units)", LongType(), True),
    StructField("Actual Baseline Volume (Units)", LongType(), True),
    StructField("Planned ROI%", DoubleType(), True),
    StructField("Actual ROI%", DoubleType(), True),
    StructField("Planned ROMI %", DoubleType(), True),
    StructField("Actual ROMI %", DoubleType(), True)
])

# Create the Spark DataFrame with selected columns
data = [
    ("A-84941", "DEOS", "EXA", "DEOS EXA DEOS - AERO BS XL - 210 ML", "TMI CAT - Plan Promo Sulley - Coppel 2x115", 12104, 170, -15.20, 14571.00, -15.20, 14571.00),
    ("A-105672", "DEOS", "EXA", "DEOS EXA DEOS - AERO BS XL - 210 ML", "DEO C1 2024 90 DIAS (ENE) - SULLY - 2x $120 - 30 DÍAS - S/FRAGUA", 1631, 105, 164.20, 4208.00, 164.20, 4208.00),
    ("A-105673", "DEOS", "EXA", "DEOS EXA DEOS - AERO BS XL - 210 ML", "DEO C1 2024 90 DIAS (ENE) - SULLY - 2x $120 - 30 DÍAS - S/FRAGUA", 162, 107, 1640.00, 4207.00, 1642.00, 208.00)
]

SalesDF = spark.createDataFrame(data, schema)

import certifi
import ssl

# ssl_context = ssl.create_default_context(cafile=certifi.where())
# chatgpt = OpenAI(model_name="gpt-3.5-turbo-1106", openai_api_kwargs={"request_timeout": 60, "ssl_context": ssl_context})



# Initialize OpenAI model
chatgpt = OpenAI(model_name="gpt-3.5-turbo-1106")

# Create the agent with a customized prompt and the PythonREPLTool
agent_executor = create_spark_dataframe_agent(chatgpt, SalesDF, additional_tools=[PythonREPLTool()], prompt="""
You are an intelligent agent tasked with analyzing a sales data DataFrame and answering questions related to promotions, volumes, and other sales metrics. The DataFrame contains columns like 'promo id', 'planned baseline volume units', 'actual baseline volume', and others. Use the provided tools and your knowledge to accurately interpret the data and provide helpful responses to the questions asked. You have access to the PySpark library and can use its functions to perform operations on the DataFrame, such as filtering, aggregating, sorting, and more. If you need to perform any data manipulation or calculations, feel free to use the Python REPL tool and leverage PySpark functions.
""")

# Define the chatbot function
def chatbot(input_text):
    if input_text:
        reply = agent_executor.run(input_text)
        return reply

# Create Gradio interface
iface = gr.Interface(
    chatbot,
    gr.components.Textbox(lines=7, label="Chat with Me"),
    gr.components.Textbox(label="Reply"),
    title="Unilever bot",
    description="Ask anything you want",
    theme="compact")

# Launch the interface with the share=True and server_name="0.0.0.0" arguments
iface.launch(share=False, server_name="0.0.0.0",server_port=8080)
#iface.launch(share=True, server_name="10.237.61.186",server_port=8080)

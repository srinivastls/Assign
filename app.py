import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os

# Setting up the api key
os.environ["OPENAI_API_KEY"] = "your-key"

def main():
    st.title("CSV Data Analysis and Query App")

    option = st.sidebar.selectbox("Select Option", ["Data Analysis", "Query"])

    if option == "Data Analysis":
        data_analysis()
    elif option == "Query":
        query_data()

def data_analysis():
    st.subheader("CSV Data Analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Select Columns for Analysis")
        columns = df.columns.tolist()
        x_column = st.selectbox("X-axis", columns)
        y_column = st.selectbox("Y-axis", columns)

        # Plot selected columns
        st.subheader("Plot")
        plot_type = st.selectbox("Select plot type", ["line", "scatter", "histogram", "boxplot", "bar"])
        plot_data(df, x_column, y_column, plot_type)

        # Statistical Analysis
        st.subheader("Statistical Analysis")
        #stats_option = st.selectbox("Select analysis type", ["Describe", "Correlation"])
        
        st.write(df[x_column].describe())
def plot_data(df, x_column, y_column, plot_type):
    if plot_type == "line":
        plt.plot(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()
    elif plot_type == "scatter":
        plt.scatter(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()
    elif plot_type == "histogram":
        plt.hist(df[y_column], bins=20)
        plt.xlabel(y_column)
        plt.ylabel("Frequency")
        st.pyplot()
    elif plot_type == "boxplot":
        sns.boxplot(x=df[x_column], y=df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()
    elif plot_type == "bar":
        bar_data = df.groupby(x_column)[y_column].sum().reset_index()
        plt.bar(bar_data[x_column], bar_data[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        st.pyplot()

def query_data():
    st.subheader("CSV Data Query")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        query = st.text_area("Insert your query")
        if st.button("Submit Query", type="primary"):
            # Create an agent from the DataFrame.
            agent = create_agent(uploaded_file)

            # Query the agent.
            response = query_agent(agent=agent, query=query)

            # Decode the response.
            decoded_response = decode_response(response)

            # Write the response to the Streamlit app.
            write_response(decoded_response)

def create_agent(filename: str):

    # Create an OpenAI object.
    llm = OpenAI(temperature=0)

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=False,handle_parsing_errors=True)

def query_agent(agent, query):


    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()


def decode_response(response: str) -> dict:
 
    return json.loads(response)

def write_response(response_dict: dict):


    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

if __name__ == "__main__":
    main()

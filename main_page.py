import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
from requests.exceptions import HTTPError, Timeout, RequestException
import pymysql
import pandas as pd
import sys
from st_aggrid import AgGrid
api_key="AIzaSyBRFnBgKg0F_Q1bdUn7zkcbe-zy7ygwlzI"
conn = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="nov@2024",
        database="projects"
    )
cursor = conn.cursor()
# Open the database connection globally at the start
#fetch books by query

def fetch_data(query, api_key, number_of_books):
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    status = {"status": "", "message": "", "http_code": None}

    if not query:
        status["status"] = "error"
        status["message"] = "Invalid search query."
        return status

    startIndex = 0
    book_data = []

    def scrap(api_key, query, startIndex):
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            "q": query,
            "key": api_key,
            "maxResults": 40,
            "startIndex": startIndex
        }

        retries = 3
        retry_delay = 2

        for attempt in range(retries):
            try:
                response = requests.get(url=url, params=params, timeout=10)
                status_code = response.status_code
                response.raise_for_status()
                data = response.json()

                if "items" in data and len(data["items"]) > 0:
                    for item in data["items"]:
                        book_id = item.get("id", None)
                        if not book_id:
                            continue

                        cursor.execute("SELECT COUNT(*) FROM book_scape_explorer WHERE book_id = %s", (book_id,))
                        exists_in_db = cursor.fetchone()[0]

                        if exists_in_db > 0 or any(book[0] == book_id for book in book_data):
                            continue

                        title = item.get("volumeInfo", {}).get("title", "Title not found")
                        subtitle = item.get("volumeInfo", {}).get("subtitle", None)
                        book_authors = ",".join(item.get("volumeInfo", {}).get("authors", []))
                        book_description = item.get("volumeInfo", {}).get("description", "Null")
                        industry_identifiers = ",".join([x.get("type", None) for x in item.get("volumeInfo", {}).get("industryIdentifiers", [])])
                        text_reading_module = item.get("volumeInfo", {}).get("readingModes", {}).get("text", None)
                        image_reading_module = item.get("volumeInfo", {}).get("readingModes", {}).get("image", None)
                        page_count = item.get("volumeInfo", {}).get("pageCount", None)
                        categories = ",".join(item.get("volumeInfo", {}).get("categories", []))
                        languages = item.get("volumeInfo", {}).get("language", None)
                        image_links_data = item.get("volumeInfo", {}).get("imageLinks", {})
                        small_thumbnail = image_links_data.get("smallThumbnail", None)
                        thumbnail = image_links_data.get("thumbnail", None)
                        image_links = ",".join([small_thumbnail, thumbnail]) if small_thumbnail and thumbnail else None
                        ratings_count = item.get("volumeInfo", {}).get("ratingsCount", None)
                        avg_rating = item.get("volumeInfo", {}).get("averageRating", None)
                        country = item.get("saleInfo", {}).get("country", None)
                        saleability = item.get("saleInfo", {}).get("saleability", None)
                        isEbook = item.get("saleInfo", {}).get("isEbook", None)
                        amount_listPrice = item.get("saleInfo", {}).get("listPrice", {}).get("amount", None)
                        currencyCode_listPrice = item.get("saleInfo", {}).get("listPrice", {}).get("currencyCode", None)
                        amount_retailPrice = item.get("saleInfo", {}).get("retailPrice", {}).get("amount", None)
                        currencyCode_retailPrice = item.get("saleInfo", {}).get("retailPrice", {}).get("currencyCode", None)
                        buyLink = item.get("volumeInfo", {}).get("infoLink", None)
                        year = item.get("volumeInfo", {}).get("publishedDate", None)
                        publisher = item.get("volumeInfo", {}).get("publisher", None)

                        book_data.append((
                            book_id, query, title, subtitle, book_authors, book_description,
                            industry_identifiers, text_reading_module, image_reading_module, page_count,
                            categories, languages, image_links, ratings_count, avg_rating,
                            country, saleability, isEbook, amount_listPrice, currencyCode_listPrice,
                            amount_retailPrice, currencyCode_retailPrice, buyLink, year, publisher
                        ))

                        if len(book_data) >= number_of_books:
                            break

                return status_code, book_data
            except (HTTPError, Timeout) as e:
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                else:
                    status["status"] = "error"
                    status["message"] = str(e)
                    status["http_code"] = response.status_code if 'response' in locals() else None
                    return status_code, []

    while len(book_data) < number_of_books:
        http_code, book_data = scrap(api_key, query, startIndex)
        if status["status"] == "error":
            return status
        startIndex += 40

    if book_data:
        try:
            cursor.executemany('''
            INSERT INTO book_scape_explorer (
                book_id, search_key, book_title, book_subtitle, book_authors, book_description,
                industryIdentifiers, text_readingModes, image_readingModes, pageCount, categories,
                languages, imageLinks, ratingsCount, averageRating, country, saleability,
                isEbook, amount_listPrice, currencyCode_listPrice, amount_retailPrice,
                currencyCode_retailPrice, buyLink, year_of_publication, publisher
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
            ''', book_data)

            conn.commit()
            status["status"] = "success"
            status["message"] = f"{cursor.rowcount} records inserted."
            status["http_code"] = http_code
        except Exception as e:
            status["status"] = "error"
            status["message"] = str(e)
        finally:
            cursor.close()
            conn.close()
    else:
        status["status"] = "error"
        status["message"] = "No book data fetched."

    return status


#display all records of the table
@st.cache_data
def displayrecords():
    global conn
    if conn.open == False:  # Check if the connection is closed
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="nov@2024",
            database="projects"
        )
    cursor = conn.cursor()
    # Write your query here to fetch the data
    query = "SELECT * FROM Book_Scape_Explorer ;"
    cursor.execute(query)
    # Fetch all rows
    rows = cursor.fetchall()
    # Get column names
    columns = [desc[0] for desc in cursor.description]
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(rows, columns=columns)
    # filtered_df=df[["book_title","search_key","year_of_publication"]]
    # filtered_df = filtered_df[filtered_df.notnull().all(axis=1)]
    # Close the database connection
    cursor.close()
    return df

def get_total_count():
    """Fetch the total count of book_id from the database."""
    global conn
    if conn.open == False:  # Reconnect if connection is closed
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="nov@2024",
            database="projects"
        )
    cursor = conn.cursor()
    query = "SELECT COUNT(book_id) FROM Book_Scape_Explorer;"
    cursor.execute(query)
    count = cursor.fetchone()[0]
    cursor.close()
    return count

def one(df):
    ebook_count = df['isEbook'].value_counts()
    ebook_count.index = ['Physical Book', 'eBook']
    plt.figure(figsize=(3, 3))
    fig, ax = plt.subplots()
    ax.pie(ebook_count, labels=ebook_count.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(ebook_count)))
    ax.axis('equal') 
    plt.title('Availability of eBooks vs Physical Books')
    st.pyplot(fig)

def two(df):
    df_cleaned = df[df['publisher'].notnull() & (df['book_authors'] != '')]
    publisher_count = df_cleaned['publisher'].value_counts()
    top_publisher = publisher_count.idxmax()  # This gives the publisher with the most books
    top_books_count = publisher_count.max()   # This gives the count of books by this publisher
    st.subheader(f"Publisher with the Most Books Published: {top_publisher}")
    st.write(f"Max Number Number of Books published by {top_publisher}: {top_books_count}")
    fig, ax = plt.subplots(figsize=(5, 3))
    publisher_count.head(10).plot(kind='bar', color=sns.color_palette("Set2", len(publisher_count)), ax=ax)
    ax.set_title("Top 10 Publishers with the Most Books Published")
    ax.set_xlabel("Publisher")
    ax.set_ylabel("Number of Books")
    st.pyplot(fig)

def three(df):
    # Clean the data: Remove null values in 'averageRating' and 'publisher'
    df_cleaned = df[df['averageRating'].notnull() & df['publisher'].notnull()]
    # Group by publisher and calculate the average rating for each publisher
    publisher_avg_rating = df_cleaned.groupby('publisher')['averageRating'].mean()
    # Filter publishers with more than 10 books
    publisher_count = df_cleaned['publisher'].value_counts()
    eligible_publishers = publisher_count[publisher_count > 10].index
    publisher_avg_rating = publisher_avg_rating.loc[eligible_publishers]
    # Find the publisher with the highest average rating
    top_publisher = publisher_avg_rating.idxmax()
    top_avg_rating = publisher_avg_rating.max()
    # Display the result in Streamlit
    st.subheader(f"Publisher with the Highest Average Rating: {top_publisher}")
    st.write(f"Highest Average Rating: {top_avg_rating:.2f}")
    # Plot the top 10 publishers by average rating
    fig, ax = plt.subplots(figsize=(8, 5))
    publisher_avg_rating.sort_values(ascending=False).head(10).plot(kind='bar', color=sns.color_palette("coolwarm", 10), ax=ax)
    ax.set_title("Top 10 Publishers with Highest Average Ratings")
    ax.set_xlabel("Publisher")
    ax.set_ylabel("Average Rating")
    plt.xticks(rotation=0)
    st.pyplot(fig)

def four(df):
    df['amount_retailPrice'] = pd.to_numeric(df['amount_retailPrice'], errors='coerce')
    df_cleaned = df.dropna(subset=['amount_retailPrice'])
    # Sort the DataFrame by 'amount_retailPrice' in descending order and take the top 5
    top_expensive_books = df_cleaned[['book_title', 'publisher', 'amount_retailPrice', 'currencyCode_retailPrice']] \
                          .sort_values(by='amount_retailPrice', ascending=False).head(5)
    # Display the top 5 most expensive books
    st.subheader("Top 5 Most Expensive Books by Retail Price")
    fig, ax = plt.subplots(figsize=(6,6))
    top_expensive_books.set_index('book_title')['amount_retailPrice'].plot(kind='bar', color=sns.color_palette("Blues", 5), ax=ax)
    ax.set_title("Top 5 Most Expensive Books by Retail Price")
    ax.set_xlabel("Book Title")
    ax.set_ylabel("Retail Price")
    ax.tick_params(axis='x', labelrotation=90)   
    fig.tight_layout()
    st.pyplot(fig)

def five(df):
    # Ensure 'year_of_publication' is numeric and 'pageCount' is also numeric
    df['year_of_publication'] = pd.to_numeric(df['year_of_publication'], errors='coerce')
    df['pageCount'] = pd.to_numeric(df['pageCount'], errors='coerce')
    df_filtered = df[(df['year_of_publication'] > 2010) & (df['pageCount'] >= 500)]
    st.subheader("Books Published After 2010 with at Least 500 Pages")
    top_books = df_filtered[['book_title', 'pageCount']].sort_values(by='pageCount', ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 8))
    top_books.set_index('book_title')['pageCount'].plot(kind='bar', color=sns.color_palette("viridis", 5), ax=ax)
    ax.set_title("Top 5 Books Published After 2010 with at Least 500 Pages")
    ax.set_xlabel("Book Title")
    ax.set_ylabel("Page Count")
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    st.pyplot(fig)

def seven(df):
    avg_page_count = df.groupby('isEbook')['pageCount'].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_page_count.plot(kind='bar', color=sns.color_palette("Set2", len(avg_page_count)), ax=ax)
    ax.set_title("Average Page Count for eBooks vs Physical Books")
    ax.set_xlabel("Book Type (eBook or Physical)")
    ax.set_ylabel("Average Page Count")
    ax.tick_params(axis='x', labelrotation=0)
    fig.tight_layout()
    st.pyplot(fig)

def eight(df):
    # Remove any rows with null or empty authors
    df_cleaned = df[df['book_authors'].notnull() & (df['book_authors'] != '')]
    # Split the authors in case there are multiple authors and count books for each author
    authors = df_cleaned['book_authors'].str.split(',').explode().str.strip()
    author_counts = authors.value_counts().head(3)
    fig, ax = plt.subplots(figsize=(8, 5))
    author_counts.plot(kind='bar', color=sns.color_palette("Set2", len(author_counts)), ax=ax)
    ax.set_title("Top 3 Authors with the Most Books")
    ax.set_xlabel("Author")
    ax.set_ylabel("Number of Books")
    ax.tick_params(axis='x', labelrotation=0)
    fig.tight_layout()
    st.pyplot(fig)

def nine(df):
    df_cleaned = df[df['publisher'].notnull() & (df['publisher'] != '')]
    publisher_count = df_cleaned['publisher'].value_counts()
    publishers_more_than_10 = publisher_count[publisher_count > 10]
    fig, ax = plt.subplots(figsize=(8, 8))
    publishers_more_than_10.plot(kind='bar', color=sns.color_palette("Set2", len(publishers_more_than_10)), ax=ax)
    ax.set_title("Publishers with More than 10 Books")
    ax.set_xlabel("Publisher")
    ax.set_ylabel("Number of Books")
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    st.pyplot(fig)

def ten(df):
    df_cleaned = df.dropna(subset=['categories', 'pageCount'])
    df_cleaned['categories'] = df_cleaned['categories'].apply(lambda x: x.split(",") if isinstance(x, str) else [])
    category_page_count = []
    for index, row in df_cleaned.iterrows():
        for category in row['categories']:
            category_page_count.append((category.strip(), row['pageCount']))
    category_page_count_df = pd.DataFrame(category_page_count, columns=["Category", "PageCount"])
    avg_page_count_by_category = category_page_count_df.groupby("Category")['PageCount'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,))
    avg_page_count_by_category.plot(kind='bar', color=sns.color_palette("Set2", len(avg_page_count_by_category)), ax=ax)
    ax.set_title("Average Page Count for Each Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Average Page Count")
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    st.pyplot(fig)

def eleven(df):
    # Clean the data by removing rows where 'book_authors' is null or empty
    df_cleaned = df.dropna(subset=['book_authors'])
    df_cleaned['author_count'] = df_cleaned['book_authors'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    books_with_more_than_3_authors = df_cleaned[df_cleaned['author_count'] > 3]
    count_books = books_with_more_than_3_authors.shape[0]
    st.subheader(f"Number of Books with More Than 3 Authors: {count_books}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=books_with_more_than_3_authors['book_title'], y=books_with_more_than_3_authors['author_count'], palette="Set2", ax=ax)
    ax.set_title("Books with More Than 3 Authors")
    ax.set_xlabel("Book Title")
    ax.set_ylabel("Number of Authors")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def twelve(df):
    df_cleaned = df.dropna(subset=['ratingsCount'])
    average_ratings_count = df_cleaned['ratingsCount'].mean()
    books_above_average_ratings = df_cleaned[df_cleaned['ratingsCount'] > average_ratings_count]
    count_books = books_above_average_ratings.shape[0]
    st.subheader(f"Number of Books with Ratings Count Greater Than the Average: {count_books}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=books_above_average_ratings['book_title'], y=books_above_average_ratings['ratingsCount'], palette="Set1", ax=ax)
    ax.set_title("Books with Ratings Count Greater Than the Average")
    ax.set_xlabel("Book Title")
    ax.set_ylabel("Ratings Count")
    plt.xticks(rotation=90)
    st.pyplot(fig)

def thirteen(df):
    # Drop rows with null values in 'book_authors' and 'year_of_publication'
    df_cleaned = df.dropna(subset=['book_authors', 'year_of_publication'])
    grouped = df_cleaned.groupby(['book_authors', 'year_of_publication']).size().reset_index(name='book_count')
    same_author_same_year = grouped[grouped['book_count'] > 1]
    results = pd.merge(same_author_same_year, df_cleaned, on=['book_authors', 'year_of_publication'], how='left')
    results = results[['book_authors', 'year_of_publication', 'book_title', 'book_count']]
    total_occurrences = same_author_same_year.shape[0]
    st.subheader(f"Number of Authors with Books Published in the Same Year: {total_occurrences}")
    st.dataframe(results)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.barplot(
    #     data=same_author_same_year, 
    #     x='book_authors', 
    #     y='book_count', 
    #     hue='year_of_publication', 
    #     dodge=True, 
    #     palette="Set2", 
    #     ax=ax
    # )
    # ax.set_title("Authors with Books Published in the Same Year")
    # ax.set_xlabel("Authors")
    # ax.set_ylabel("Number of Books")
    # plt.xticks(rotation=90)
    # st.pyplot(fig)

def fourteen(df):
    # Create a text input box for the user to enter a keyword
    keyword = st.text_input("Enter a keyword to search for in book titles:", value="")
    
    # Filter books containing the keyword in their title (case insensitive)
    if keyword:
        filtered_books = df[df['book_title'].str.contains(keyword, case=False, na=False)]
        st.subheader(f"Books Containing '{keyword}' in the Title")
        st.dataframe(filtered_books[['book_title', 'book_authors', 'year_of_publication']])
        
        # Display the total count
        st.write(f"Total Books Found: {filtered_books.shape[0]}")
    else:
        st.write("Please enter a keyword to see the results.")

def fifteen(df):
    # Clean the data: Drop rows with missing year or price
    df_cleaned = df.dropna(subset=['year_of_publication', 'amount_retailPrice'])
    
    # Calculate average price per year
    avg_price_per_year = (
        df_cleaned.groupby('year_of_publication')['amount_retailPrice']
        .mean()
        .reset_index(name='average_price')
    )
    
    # Sort by average price in descending order and select the top 5
    top_5_avg_prices = avg_price_per_year.sort_values(by='average_price', ascending=False).head(5)
    
    # Display the year with the highest average book price
    max_avg_price = top_5_avg_prices.iloc[0]['average_price']
    year_with_highest_avg = top_5_avg_prices.iloc[0]['year_of_publication']
    
    st.subheader(f"Year with the Highest Average Book Price: {year_with_highest_avg}")
    st.write(f"The highest average price is **${max_avg_price:.2f}**.")
    
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_5_avg_prices, 
        x='year_of_publication', 
        y='average_price', 
        palette="viridis"
    )
    plt.title("Top 5 Years with Highest Average Book Prices")
    plt.xlabel("Year of Publication")
    plt.ylabel("Average Price ($)")
    plt.xticks(rotation=90)
    st.pyplot(plt)

def sixteen(df):
    # Clean the data: Drop rows with missing authors or publication_year
    df_cleaned = df.dropna(subset=['book_authors', 'year_of_publication'])
    
    # Group by authors and years of publication
    grouped = df_cleaned.groupby(['book_authors', 'year_of_publication']).size().reset_index(name='book_count')
    
    # Sort by authors and year of publication
    grouped = grouped.sort_values(by=['book_authors', 'year_of_publication'])
    
    # Find authors who published in 3 consecutive years
    authors_with_consecutive_years = []
    
    for author, group in grouped.groupby('book_authors'):
        years = group['year_of_publication'].values
        # Check for three consecutive years
        for i in range(len(years) - 2):
            if years[i+1] == years[i] + 1 and years[i+2] == years[i] + 2:
                authors_with_consecutive_years.append(author)
                break  # No need to check further for this author
    
    # Count how many authors have books in three consecutive years
    consecutive_authors_count = len(set(authors_with_consecutive_years))
    
    st.subheader(f"Count of Authors Who Published in 3 Consecutive Years: {consecutive_authors_count}")
    st.write(f"Total authors with books published in 3 consecutive years: **{consecutive_authors_count}**.")
    
    # Optionally, list the authors who have books in 3 consecutive years
    st.subheader("Authors Who Published in 3 Consecutive Years")
    authors_list = list(set(authors_with_consecutive_years))
    st.write(authors_list)

def seventeen(df):
    # Check if the required columns exist
    required_columns = ['book_authors', 'year_of_publication', 'publisher']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the dataset.")
            return
    
    # Clean the data: Drop rows with missing authors, publishers, or publication year
    df_cleaned = df.dropna(subset=required_columns)
    
    # Group by authors and year of publication, and aggregate unique publishers
    grouped = df_cleaned.groupby(['book_authors', 'year_of_publication'])['publisher'].nunique().reset_index()
    
    # Filter authors who have more than one publisher in the same year
    authors_with_multiple_publishers = grouped[grouped['publisher'] > 1]
    
    if authors_with_multiple_publishers.empty:
        st.subheader("No authors found with books published in the same year under different publishers.")
        return
    
    # Get the authors who published in the same year under different publishers
    result = pd.merge(authors_with_multiple_publishers, df_cleaned, on=['book_authors', 'year_of_publication'], how='left')
    
    # Display the result
    st.subheader("Authors Who Published Books in the Same Year But Under Different Publishers")
    st.write("The following authors have published books in the same year with different publishers:")
    st.dataframe(result[['book_authors', 'year_of_publication', 'publisher', 'book_title']])


    # Check if the required columns exist
    required_columns = ['isEbook', 'amount_retailPrice']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column '{col}' is missing from the dataset.")
            return
    
    # Clean the data: Drop rows with missing values for isEbook or amount_retailPrice
    df_cleaned = df.dropna(subset=required_columns)
    
    # Group by isEbook and calculate the average price
    avg_price = df_cleaned.groupby('isEbook')['amount_retailPrice'].mean().reset_index()
    avg_price['isEbook'] = avg_price['isEbook'].map({1: 'eBook', 0: 'Physical Book'})
    
    # Display the results
    st.subheader("Average Retail Price of eBooks vs Physical Books")
    st.dataframe(avg_price.rename(columns={'isEbook': 'Book Type', 'amount_retailPrice': 'Average Price ($)'}))
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=avg_price, x='isEbook', y='amount_retailPrice', palette="Set2", ax=ax)
    ax.set_title("Average Retail Price: eBooks vs Physical Books")
    ax.set_xlabel("Book Type")
    ax.set_ylabel("Average Price ($)")
    st.pyplot(fig)


    # Clean the data: Drop rows with missing values for isEbook or amount_retailPrice
    df_cleaned = df.dropna(subset=['isEbook', 'amount_retailPrice'])
    
    # Group by isEbook and calculate the average price
    avg_price = df_cleaned.groupby('isEbook')['amount_retailPrice'].mean().reset_index()
    avg_price['Book Type'] = avg_price['isEbook'].map({1: 'eBook', 0: 'Physical Book'})
    
    # Prepare data for visualization
    avg_price_values = avg_price['amount_retailPrice']
    avg_price_labels = avg_price['Book Type']
    
    # Plot the pie chart
    plt.figure(figsize=(3, 3))
    fig, ax = plt.subplots()
    ax.pie(
        avg_price_values,
        labels=avg_price_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("Set2", len(avg_price_values))
    )
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    plt.title('Average Retail Price: eBooks vs Physical Books')
    st.pyplot(fig)
    
    # Display numerical values
    st.subheader("Average Retail Price of eBooks vs Physical Books")
    st.dataframe(avg_price[['Book Type', 'amount_retailPrice']].rename(columns={'amount_retailPrice': 'Average Price ($)'}))

def eighteen(df):
    df_cleaned = df.dropna(subset=['isEbook', 'amount_retailPrice'])
    avg_price = df_cleaned.groupby('isEbook')['amount_retailPrice'].mean().reset_index()
    avg_price = avg_price.set_index('isEbook').reindex([1, 0], fill_value=0).reset_index()
    avg_price['Book Type'] = avg_price['isEbook'].map({1: 'eBook', 0: 'Physical Book'})
    ebook_avg_price = avg_price.loc[avg_price['Book Type'] == 'eBook', 'amount_retailPrice'].values[0]
    physical_avg_price = avg_price.loc[avg_price['Book Type'] == 'Physical Book', 'amount_retailPrice'].values[0]
    st.write(f"Average Price of eBooks: **${ebook_avg_price:.2f}**")
    st.write(f"Average Price of Physical Books: **${physical_avg_price:.2f}**")
    fig, ax = plt.subplots(figsize=(2,2))
    sns.barplot(
        data=avg_price,
        x='Book Type',
        y='amount_retailPrice',
        palette='Set2',
        ax=ax
    )
    ax.set_title('Avg Price: eBooks vs Physical Books', fontsize=10)
    ax.set_ylabel('Avg Price ($)', fontsize=5)
    ax.set_xlabel('Book Type', fontsize=5)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    st.pyplot(fig)

def nineteen(df):

    df_cleaned = df.dropna(subset=['averageRating'])
    df_cleaned['averageRating'] = df_cleaned['averageRating'].astype(float)
    mean_rating = df_cleaned['averageRating'].mean()
    std_dev_rating = df_cleaned['averageRating'].std()
    lower_bound = mean_rating - 2 * std_dev_rating
    upper_bound = mean_rating + 2 * std_dev_rating
    outliers = df_cleaned[(df_cleaned['averageRating'] < lower_bound) | (df_cleaned['averageRating'] > upper_bound)]
    st.subheader("Books with Ratings More Than Two Standard Deviations Away from the Average Rating")
    st.write(f"The average rating of all books is **{mean_rating:.2f}** and the standard deviation is **{std_dev_rating:.2f}**.")
    st.dataframe(outliers[['book_title', 'book_authors', 'averageRating']])
    plt.figure(figsize=(8, 6))
    sns.barplot(data=outliers, x='book_title', y='averageRating', palette='Set2')
    plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels for readability
    plt.xlabel('Book Title')
    plt.ylabel('Average Rating')
    plt.title('Books with Ratings More Than Two Standard Deviations from the Average Rating')
    plt.tight_layout()  # Ensure that the plot fits within the window
    st.pyplot(plt)


    df_cleaned = df.dropna(subset=['publisher', 'averageRating'])
    publisher_counts = df_cleaned.groupby('publisher').size()
    publishers_with_more_than_10_books = publisher_counts[publisher_counts > 10].index
    filtered_df = df_cleaned[df_cleaned['publisher'].isin(publishers_with_more_than_10_books)]
    avg_ratings_per_publisher = filtered_df.groupby('publisher')['averageRating'].mean().reset_index()
    top_publisher = avg_ratings_per_publisher.loc[avg_ratings_per_publisher['averageRating'].idxmax()]
    st.subheader("Publisher with the Highest Average Rating (for Publishers with More Than 10 Books)")
    st.write(f"The publisher with the highest average rating is: {top_publisher['publisher']} with an average rating of {top_publisher['averageRating']:.2f}")

def twenty(df):
    # Clean the data: Drop rows with missing publisher or averageRating
    df_cleaned = df.dropna(subset=['publisher', 'averageRating'])
    publisher_counts = df_cleaned.groupby('publisher').size()
    publishers_with_more_than_10_books = publisher_counts[publisher_counts > 10].index
    filtered_df = df_cleaned[df_cleaned['publisher'].isin(publishers_with_more_than_10_books)]
    avg_ratings_per_publisher = filtered_df.groupby('publisher')['averageRating'].mean().reset_index()
    avg_ratings_per_publisher = avg_ratings_per_publisher.sort_values('averageRating', ascending=False).reset_index(drop=True)
    st.subheader("Publishers with More Than 10 Books and Their Average Ratings")
    st.dataframe(avg_ratings_per_publisher)


with st.sidebar:
    selected = option_menu(
        menu_title="Home",  # Sidebar title
        options=["Info", "Extract Books", "Data Analysis"],  # Page options
        icons=["info-circle", "book", "bar-chart"],  # Icons for each page
        menu_icon="cast",  # Icon for the sidebar
        default_index=0,  # Default selected page
    )

# Display content based on selected page
if selected == "Info":
    st.markdown(
        """
        <h1 style="text-align: center;">Book Scape Explorer</h1>
        <p style="text-align: center;font-size: 24px">The BookScape Explorer project aims to facilitate users in discovering and analyzing book data through a web application. The application extracts data from online book APIs, stores this information in a SQL database, and enables data analysis using SQL queries. The project will provide insights into book trends, user preferences, and reviews, helping users make informed reading choices while also offering a platform for community engagement. This initiative targets avid readers, researchers, and book enthusiasts.</p>
        """,
        unsafe_allow_html=True,
    )

elif selected == "Extract Books":
    st.title("Feed books into the DB")
    if "df" not in st.session_state:
        st.session_state.df = displayrecords()  # Load records into session state
    query = st.text_input("Search the book you want to enter")
    selected_category = st.selectbox("Choose a category", [100, 200, 300])
    # Create two columns to display buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch", key="fetch_button"):
            # Call fetch_data and capture the status
            status = fetch_data(query, api_key, selected_category)
            if status["status"] == "success":
                # Refresh the displayed records and count
                st.session_state.df = displayrecords()
                length = get_total_count()
                st.success(status["message"])
                st.info(f"Number of records in the table: {length}")
            elif status["status"] == "error":
                # Show error messages
                st.error(f"Error: {status['message']}")
                if status["http_code"]:
                    st.info(f"HTTP Code: {status['http_code']}")
    # Button 2 (Refresh Data)
    with col2:
        if st.button("Refresh Data", key="refresh_data"):
            st.session_state.df = displayrecords()  # Refresh records
            st.success("Data refreshed successfully!")
    AgGrid(st.session_state.df)  # Show the table
    length = get_total_count()
    st.info(f"Number of records in the table: {length}")

          
elif selected == "Data Analysis":
    df=displayrecords()
    st.header("Insights and Trends from the Book Database")
    questions = [
        "1.Availability of eBooks vs Physical Books",
        "2.Publisher with the Most Books Published",
        "3.Publisher with the Highest Average Rating",
        "4.Top 5 Most Expensive Books by Retail Price",
        "5.Books Published After 2010 with at Least 500 Pages",
        "6.Books with Discounts Greater than 20%",
        "7.Average Page Count for eBooks vs Physical Books",
        "8.Top 3 Authors with the Most Books",
        "9.Publishers with More than 10 Books",
        "10.Average Page Count for Each Category",
        "11.Books with More than 3 Authors",
        "12.Books with Ratings Count Greater Than the Average",
        "13.Books with the Same Author Published in the Same Year",
        "14.Books with a Specific Keyword in the Title",
        "15.Year with the Highest Average Book Price",
        "16.Count of Authors Who Published 3 Consecutive Years",
        "17.authors who have published books in the same year but under different publishers.",
        "18.the average amount_retailPrice of eBooks and physical books.",
        "19.books that have an averageRating that is more than two standard deviations away from the average rating of all books.",
        "20.publisher which has the highest average rating among its books, for publishers that have published more than 10 books."
    ]
    selected_category = st.selectbox("Choose a category", questions)
    if selected_category == "1.Availability of eBooks vs Physical Books":
        one(df)
    elif selected_category=="2.Publisher with the Most Books Published":
        two(df)
    elif selected_category=="3.Publisher with the Highest Average Rating":
        three(df)
    elif selected_category=="4.Top 5 Most Expensive Books by Retail Price":
        four(df)
    elif selected_category=="5.Books Published After 2010 with at Least 500 Pages":
        five(df)
    elif selected_category=="7.Average Page Count for eBooks vs Physical Books":
        seven(df)
    elif selected_category=="8.Top 3 Authors with the Most Books":
        eight(df)
    elif selected_category=="9.Publishers with More than 10 Books":
        nine(df)
    elif selected_category=="10.Average Page Count for Each Category":
        ten(df)
    elif selected_category=="11.Books with More than 3 Authors":
        eleven(df)
    elif selected_category=="12.Books with Ratings Count Greater Than the Average":
        twelve(df)
    elif selected_category=="13.Books with the Same Author Published in the Same Year":
        thirteen(df) 
    elif selected_category=="14.Books with a Specific Keyword in the Title":
        fourteen(df) 
    elif selected_category=="15.Year with the Highest Average Book Price":
        fifteen(df)
    elif selected_category=="16.Count of Authors Who Published 3 Consecutive Years":
        sixteen(df)
    elif selected_category=="17.authors who have published books in the same year but under different publishers.":
        seventeen(df)
    elif selected_category=="18.the average amount_retailPrice of eBooks and physical books.":
        eighteen(df)
    elif selected_category=="19.books that have an averageRating that is more than two standard deviations away from the average rating of all books.":
        nineteen(df)
    elif selected_category=="20.publisher which has the highest average rating among its books, for publishers that have published more than 10 books.":
        twenty(df)
        
        
        
        
    
        
        




# Close the connection at the end of the app if necessary
# conn.close()  # Uncomment this line if you want to close the connection manually when the app stops


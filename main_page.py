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
from st_aggrid import AgGrid,GridOptionsBuilder
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
    if not conn.open:  # Check if the connection is closed
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="nov@2024",
            database="projects"
        )

    cursor = conn.cursor()
    query = """
SELECT book_authors, book_title, year_of_publication
    FROM Book_Scape_Explorer
    WHERE book_authors IS NOT NULL 
      AND book_authors != '' 
      AND book_title IS NOT NULL 
      AND book_title != ''
      AND year_of_publication IS NOT NULL
      AND book_title != '';
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    df.rename(columns={
        "book_authors": "Authors",
        "book_title": "BOOK",
        "year_of_publication": "YEAR PUBLISHED"
    }, inplace=True)
    df.dropna(inplace=True)
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

def one():
    global conn
    try:
        # Reconnect if the connection is closed
        if not conn.open:  
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )
        cursor = conn.cursor()
        query = """
        SELECT 
            CASE 
                WHEN isEbook = TRUE THEN 'eBook' 
                ELSE 'Physical Book' 
            END AS book_type, 
            COUNT(*) AS count 
        FROM book_scape_explorer 
        GROUP BY book_type;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        # Extract data for the pie chart
        labels, sizes = zip(*rows)

        st.subheader("Number of eBooks and Physical Books")
        for label, size in zip(labels, sizes):
            st.write(f"{label}: {size}")

        # Create pie chart
        plt.figure(figsize=(6, 6))
        fig, ax = plt.subplots()
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(labels))
        )
        ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
        plt.title('Distribution of Book Types')
        
        # Display the pie chart in Streamlit
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def two():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )
        cursor = conn.cursor()
        query = """
        SELECT 
            publisher, 
            COUNT(*) AS book_count
        FROM book_scape_explorer
        WHERE publisher IS NOT NULL AND book_authors != ''
        GROUP BY publisher
        ORDER BY book_count DESC
        LIMIT 10;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        top_publisher = df.iloc[0]['publisher']
        top_books_count = df.iloc[0]['book_count']
        st.subheader(f"Publisher with the Most Books Published: {top_publisher}")
        st.write(f"Max Number of Books published by {top_publisher}: {top_books_count}")
        fig, ax = plt.subplots(figsize=(5, 3))
        df.set_index('publisher')['book_count'].plot(
            kind='bar', 
            color=sns.color_palette("Set2", len(df)), 
            ax=ax
        )
        ax.set_title("Top 10 Publishers with the Most Books Published")
        ax.set_xlabel("Publisher")
        ax.set_ylabel("Number of Books")
        st.pyplot(fig)
    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()
 
def three():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )
        cursor = conn.cursor()
        query = """
        SELECT 
            publisher, 
            AVG(averageRating) AS avg_rating
        FROM book_scape_explorer
        WHERE averageRating IS NOT NULL AND publisher IS NOT NULL
        GROUP BY publisher
        HAVING COUNT(*) > 10
        ORDER BY avg_rating DESC
        LIMIT 10;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
        if df.empty:
            st.warning("No data available for publishers with the highest average rating.")
            return
        top_publisher = df.iloc[0]['publisher']
        top_avg_rating = df.iloc[0]['avg_rating']
        st.subheader(f"Publisher with the Highest Average Rating: {top_publisher}")
        st.write(f"Highest Average Rating: {top_avg_rating:.2f}")
        fig, ax = plt.subplots(figsize=(8, 5))
        df.set_index('publisher')['avg_rating'].sort_values(ascending=False).plot(
            kind='bar', 
            color=sns.color_palette("coolwarm", len(df)), 
            ax=ax
        )
        ax.set_title("Top 10 Publishers with Highest Average Ratings")
        ax.set_xlabel("Publisher")
        ax.set_ylabel("Average Rating")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        # Ensure the cursor is closed
        cursor.close()
  
def four():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT 
            book_title, 
            publisher, 
            amount_retailPrice, 
            currencyCode_retailPrice
        FROM book_scape_explorer
        WHERE amount_retailPrice IS NOT NULL
        ORDER BY amount_retailPrice DESC
        LIMIT 5;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            st.warning("No data available for the most expensive books.")
            return
        df['amount_retailPrice'] = pd.to_numeric(df['amount_retailPrice'], errors='coerce')
        if df['amount_retailPrice'].isnull().all():
            st.warning("No valid numeric data for the most expensive books.")
            return
        # Get the details of the most expensive book
        most_expensive_book = df.iloc[0]
        book_title = most_expensive_book['book_title']
        retail_price = most_expensive_book['amount_retailPrice']
        currency = most_expensive_book['currencyCode_retailPrice']
        st.subheader("Top 5 Most Expensive Books by Retail Price")
        st.write(f"**Most Expensive Book:** {book_title}")
        st.write(f"**Retail Price:** {retail_price} {currency}")
        fig, ax = plt.subplots(figsize=(6, 6))
        df.set_index('book_title')['amount_retailPrice'].plot(
            kind='bar', 
            color=sns.color_palette("Blues", len(df)), 
            ax=ax
        )
        ax.set_title("Top 5 Most Expensive Books by Retail Price")
        ax.set_xlabel("Book Title")
        ax.set_ylabel("Retail Price")
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def five():
    global conn
    try:
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT 
            book_title, 
            year_of_publication, 
            pageCount
        FROM book_scape_explorer
        WHERE year_of_publication > 2010 
            AND pageCount >= 500
        ORDER BY pageCount DESC
        LIMIT 5;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        st.subheader("Top 5 Books Published After 2010 with at Least 500 Pages")
        for _, row in df.iterrows():
            st.write(f"**{row['book_title']}**: {row['pageCount']} pages (Published: {row['year_of_publication']})")
        if df.empty:
            st.warning("No books found published after 2010 with at least 500 pages.")
            return
        df['year_of_publication'] = pd.to_numeric(df['year_of_publication'], errors='coerce')
        df['pageCount'] = pd.to_numeric(df['pageCount'], errors='coerce')
        fig, ax = plt.subplots(figsize=(12,12))
        df.set_index('book_title')['pageCount'].plot(
            kind='bar', 
            color=sns.color_palette("viridis", 5), 
            ax=ax
        )
        ax.set_title("Top 5 Books Published After 2010 with at Least 500 Pages")
        ax.set_xlabel("Book Title")
        ax.set_ylabel("Page Count")
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def six():
    global conn
    try:
        # Ensure database connection
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()

        # SQL query to get books with a discount greater than 20%
        query = """
        SELECT 
            book_title,
            amount_listPrice,
            amount_retailPrice,
            (amount_listPrice - amount_retailPrice) / amount_listPrice * 100 AS discount_percentage
        FROM book_scape_explorer
        WHERE (amount_listPrice - amount_retailPrice) / amount_listPrice * 100 > 20
        ORDER BY discount_percentage DESC;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            st.warning("No books found with discounts greater than 20%.")
            return

        # Convert to DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        # Display the DataFrame using AG-Grid
        st.subheader("Books with Discounts Greater than 20%")
        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_pagination(paginationPageSize=10)  # Enable pagination
        grid_options.configure_column("discount_percentage", type=["numericColumn", "editableColumn"])  # Make the discount editable
        grid_options.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
        AgGrid(df, gridOptions=grid_options.build(), height=400, width='100%')
    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def seven():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT 
            isEbook, 
            AVG(CASE WHEN pageCount IS NOT NULL AND pageCount > 0 THEN pageCount ELSE NULL END) AS avg_page_count
        FROM book_scape_explorer
        GROUP BY isEbook;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # Convert SQL results to DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        if df.empty:
            st.warning("No data available for average page count comparison.")
            return
        # Ensure that 'avg_page_count' is numeric
        df['avg_page_count'] = pd.to_numeric(df['avg_page_count'], errors='coerce')

        # Debugging: Check if the 'avg_page_count' column has valid numeric data
        if df['avg_page_count'].isnull().all():
            st.warning("No valid numeric data available for average page count.")
            return
        
        # Map 'isEbook' values: 1 to "eBook" and 0 to "Physical Book"
        df['isEbook'] = df['isEbook'].replace({1: 'eBook', 0: 'Physical Book'})
        
        st.subheader("Average Page Count for eBooks vs Physical Books")
        
        # Plotting the average page count for eBooks and Physical Books
        fig, ax = plt.subplots(figsize=(6, 4))
        df.set_index('isEbook')['avg_page_count'].plot(
            kind='bar', 
            color=sns.color_palette("Set2", len(df)), 
            ax=ax
        )
        ax.set_title("Average Page Count for eBooks vs Physical Books")
        ax.set_xlabel("Book Type (eBook or Physical)")
        ax.set_ylabel("Average Page Count")
        ax.tick_params(axis='x', labelrotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def eight():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()

        # SQL query to get the top 3 authors with the most books
        query = """
        SELECT
            TRIM(author) AS author,
            COUNT(*) AS book_count
        FROM (
            SELECT
                TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(book_authors, ',', n.n), ',', -1)) AS author
            FROM
                book_scape_explorer
            CROSS JOIN
                (SELECT 1 AS n UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10) n
            WHERE
                book_authors IS NOT NULL AND book_authors != ''
                AND CHAR_LENGTH(book_authors) - CHAR_LENGTH(REPLACE(book_authors, ',', '')) >= n.n - 1
        ) AS authors
        GROUP BY author
        ORDER BY book_count DESC
        LIMIT 3;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            st.warning("No data available for the top 3 authors with the most books.")
            return

        # Convert SQL results to DataFrame
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        # Display the authors and book counts as plain text
        st.subheader("Top 3 Authors with the Most Books")
        for _, row in df.iterrows():
            st.write(f"**Author:** {row['author']} - **Books Count:** {row['book_count']}")

        # Plotting the top 3 authors
        fig, ax = plt.subplots(figsize=(8, 5))
        df.set_index('author')['book_count'].plot(
            kind='bar', 
            color=sns.color_palette("Set2", len(df)), 
            ax=ax
        )
        ax.set_title("Top 3 Authors with the Most Books")
        ax.set_xlabel("Author")
        ax.set_ylabel("Number of Books")
        ax.tick_params(axis='x', labelrotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def nine():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )
        cursor = conn.cursor()
        query = """
        SELECT
            publisher,
            COUNT(*) AS book_count
        FROM
            book_scape_explorer
        WHERE
            publisher IS NOT NULL AND publisher != ''
        GROUP BY
            publisher
        HAVING
            COUNT(*) > 10
        ORDER BY
            book_count DESC;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            st.warning("No publishers with more than 10 books found.")
            return
        st.write("Retrieved Data:")
        st.write(df)
        st.subheader("Publishers with More than 10 Books")
        fig, ax = plt.subplots(figsize=(8, 8))
        df.set_index('publisher')['book_count'].plot(
            kind='bar', 
            color=sns.color_palette("Set2", len(df)), 
            ax=ax
        )
        ax.set_title("Publishers with More than 10 Books")
        ax.set_xlabel("Publisher")
        ax.set_ylabel("Number of Books")
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        st.pyplot(fig)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def ten():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT
            TRIM(category) AS category,
            AVG(pageCount) AS avg_page_count
        FROM (
            SELECT
                TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(categories, ',', n.n), ',', -1)) AS category,
                pageCount
            FROM
                book_scape_explorer
            CROSS JOIN
                (SELECT 1 AS n UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10) n
            WHERE
                categories IS NOT NULL AND categories != ''
                AND CHAR_LENGTH(categories) - CHAR_LENGTH(REPLACE(categories, ',', '')) >= n.n - 1
                AND pageCount IS NOT NULL AND pageCount != ''
        ) AS category_page_count
        GROUP BY category
        ORDER BY avg_page_count DESC
        LIMIT 10;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            st.warning("No data available for average page count by category.")
            return
        df['avg_page_count'] = pd.to_numeric(df['avg_page_count'], errors='coerce')
        st.write("Retrieved Data:")
        st.write(df)
        if df['avg_page_count'].isnull().all():
            st.warning("No valid numeric data for average page count by category.")
            return
        st.subheader("Average Page Count for Each Category")
        fig, ax = plt.subplots(figsize=(8, 5))
        df.set_index('category')['avg_page_count'].plot(
            kind='bar', 
            color=sns.color_palette("Set2", len(df)), 
            ax=ax
        )
        ax.set_title("Average Page Count for Each Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Average Page Count")
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        st.pyplot(fig)
    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def eleven():
    global conn
    try:
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT
            book_title,
            LENGTH(book_authors) - LENGTH(REPLACE(book_authors, ',', '')) + 1 AS author_count
        FROM
            book_scape_explorer
        WHERE
            book_authors IS NOT NULL AND book_authors != ''
        HAVING author_count > 3;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # Convert SQL results to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        if df.empty:
            st.warning("No books found with more than 3 authors.")
            return

        # Displaying the number of books with more than 3 authors and average number of authors
        book_count = df.shape[0]
        avg_authors = df['author_count'].mean()
        st.subheader(f"Number of Books with More Than 3 Authors: {book_count}")
        st.subheader(f"Average Number of Authors: {avg_authors:.2f}")

        # Setting up pagination in AG-Grid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationPageSize=10)  # Number of rows per page
        grid_options = gb.build()

        # Displaying the result in an AG-Grid table with pagination
        st.write("Books with More Than 3 Authors:")
        AgGrid(df, gridOptions=grid_options)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def twelve():
    global conn
    try:
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT
            book_title,
            ratingsCount
        FROM
            book_scape_explorer
        WHERE
            ratingsCount IS NOT NULL
        HAVING ratingsCount > (
            SELECT AVG(ratingsCount) FROM book_scape_explorer WHERE ratingsCount IS NOT NULL
        );
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            st.warning("No books found with ratings count greater than the average.")
            return
        book_count = df.shape[0]
        st.subheader(f"Number of Books with Ratings Count Greater Than the Average: {book_count}")
        st.write("Books with Ratings Count Greater Than the Average:")
        st.write(df)

    except pymysql.err.ProgrammingError as e:
        st.error(f"SQL Error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    finally:
        cursor.close()

def thirteen(df):
    df_cleaned = df.dropna(subset=['book_authors', 'year_of_publication'])
    grouped = df_cleaned.groupby(['book_authors', 'year_of_publication']).size().reset_index(name='book_count')
    same_author_same_year = grouped[grouped['book_count'] > 1]
    results = pd.merge(same_author_same_year, df_cleaned, on=['book_authors', 'year_of_publication'], how='left')
    results = results.drop(columns=['book_authors'])
    results = results[['year_of_publication', 'book_title', 'book_count']]
    total_occurrences = same_author_same_year.shape[0]
    st.subheader(f"Number of Authors with Books Published in the Same Year: {total_occurrences}")
    st.dataframe(results)

def fourteen(keyword):
    global conn
    if not conn.open:  # Check if the connection is closed
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="nov@2024",
            database="projects"
        )

    cursor = conn.cursor()
    # Use parameterized query to avoid SQL injection
    query = """
    SELECT book_authors, book_title, year_of_publication
    FROM Book_Scape_Explorer
    WHERE book_title LIKE %s
      AND book_authors IS NOT NULL
      AND book_title IS NOT NULL
      AND year_of_publication IS NOT NULL
      AND TRIM(book_authors) != ''
      AND TRIM(book_title) != ''
      AND TRIM(year_of_publication) != '';
    """
    cursor.execute(query, ('%' + keyword + '%',))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()

    # Rename the columns
    df.rename(columns={
        "book_authors": "Authors",
        "book_title": "BOOK",
        "year_of_publication": "YEAR PUBLISHED"
    }, inplace=True)

    return df


def fifteen():
    global conn
    try:
        # Reconnect if connection is closed
        if not conn.open:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="nov@2024", database="projects")
        cursor = conn.cursor()

        # SQL query for year with highest average book price
        query = """
        SELECT year_of_publication, AVG(amount_retailPrice) AS average_price
        FROM book_scape_explorer
        WHERE year_of_publication IS NOT NULL AND amount_retailPrice IS NOT NULL
        GROUP BY year_of_publication
        ORDER BY average_price DESC
        LIMIT 1;
        """
        cursor.execute(query)
        row = cursor.fetchone()
        if row:
            st.subheader(f"Year with Highest Average Price: {row[0]}")
            st.write(f"Price: **${row[1]:.2f}**")
        else:
            st.warning("No data available.")

        # Query for top 5 years
        query_top_5 = """
        SELECT year_of_publication, AVG(amount_retailPrice) AS average_price
        FROM book_scape_explorer
        WHERE year_of_publication IS NOT NULL AND amount_retailPrice IS NOT NULL
        GROUP BY year_of_publication
        ORDER BY average_price DESC
        LIMIT 5;
        """
        cursor.execute(query_top_5)
        df = pd.DataFrame(cursor.fetchall(), columns=["Year", "Average Price"])
        if not df.empty:
            st.write("Top 5 Years with Highest Average Prices:")
            st.write(df)
        else:
            st.warning("No data available for top 5 years.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()

def sixteen():
    global conn
    try:
        # Reconnect if connection is closed
        if not conn.open:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="nov@2024", database="projects")
        cursor = conn.cursor()

        # SQL query to fetch authors with books in 3 consecutive years
        query = """
        SELECT DISTINCT t1.book_authors
        FROM book_scape_explorer t1
        JOIN book_scape_explorer t2
          ON t1.book_authors = t2.book_authors
         AND t2.year_of_publication = t1.year_of_publication + 1
        JOIN book_scape_explorer t3
          ON t1.book_authors = t3.book_authors
         AND t3.year_of_publication = t1.year_of_publication + 2
        WHERE t1.book_authors IS NOT NULL
          AND t1.year_of_publication IS NOT NULL
          AND t2.year_of_publication IS NOT NULL
          AND t3.year_of_publication IS NOT NULL;
        """
        cursor.execute(query)
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Check if rows are returned
        if not rows:
            st.subheader("No authors found with books published in 3 consecutive years.")
            return
        
        # Convert the rows into a DataFrame
        df = pd.DataFrame(rows, columns=['book_authors'])
        df = df[df['book_authors'] != '']
        st.subheader("Authors Who Published Books in 3 Consecutive Years")
        st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    
def seventeen():
    global conn
    try:
        # Reconnect if connection is closed
        if not conn.open:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="nov@2024", database="projects")
        cursor = conn.cursor()

        # SQL query
        query = """
        SELECT DISTINCT 
            t1.book_authors, 
            t1.year_of_publication, 
            t1.publisher AS publisher_1, 
            t2.publisher AS publisher_2
        FROM book_scape_explorer t1
        JOIN book_scape_explorer t2
          ON t1.book_authors = t2.book_authors
         AND t1.year_of_publication = t2.year_of_publication
        WHERE t1.publisher <> t2.publisher
          AND t1.book_authors IS NOT NULL
          AND t1.year_of_publication IS NOT NULL
          AND t1.publisher IS NOT NULL
          AND t2.publisher IS NOT NULL;
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        # Display the results
        st.subheader("Authors Who Published Books in the Same Year But Under Different Publishers")
        if not df.empty:
            st.dataframe(df.rename(columns={
                'book_authors': 'Author', 
                'year_of_publication': 'Year of Publication',
                'publisher_1': 'Publisher 1',
                'publisher_2': 'Publisher 2'
            }))
        else:
            st.write("No authors found with books published in the same year under different publishers.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

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

def nineteen():
    global conn
    try:
        # Reconnect to the database if the connection is closed
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()

        # SQL query to calculate the average and standard deviation
        query = """
        SELECT
            AVG(averageRating) AS mean_rating,
            STDDEV(averageRating) AS std_dev_rating
        FROM
            book_scape_explorer
        WHERE
            averageRating IS NOT NULL;
        """
        cursor.execute(query)
        stats = cursor.fetchone()

        if not stats:
            st.warning("No valid rating data available.")
            return

        # Convert the result to float to avoid the type error
        mean_rating = float(stats[0])  # Convert to float
        std_dev_rating = float(stats[1])  # Convert to float

        # Query to get books more than 2 standard deviations away from the mean
        query_outliers = f"""
        SELECT
            book_title,
            book_authors,
            averageRating
        FROM
            book_scape_explorer
        WHERE
            averageRating < {mean_rating - 2 * std_dev_rating}
            OR averageRating > {mean_rating + 2 * std_dev_rating}
            AND averageRating IS NOT NULL;
        """

        cursor.execute(query_outliers)
        outliers = cursor.fetchall()

        # If no outliers found
        if not outliers:
            st.warning("No books found with ratings more than two standard deviations from the average.")
            return

        # Create a DataFrame for display
        df_outliers = pd.DataFrame(outliers, columns=["book_title", "book_authors", "averageRating"])

        # Display the results
        st.subheader("Books with Ratings More Than Two Standard Deviations Away from the Average Rating")
        st.write(f"The average rating of all books is **{mean_rating:.2f}** and the standard deviation is **{std_dev_rating:.2f}**.")
        st.dataframe(df_outliers[['book_title', 'book_authors', 'averageRating']])

        # Plotting the results
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_outliers, x='book_title', y='averageRating', palette='Set2')
        plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels for readability
        plt.xlabel('Book Title')
        plt.ylabel('Average Rating')
        plt.title('Books with Ratings More Than Two Standard Deviations from the Average Rating')
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()

def twenty():
    global conn
    try:
        if not conn.open:
            conn = pymysql.connect(
                host="localhost",
                port=3306,
                user="root",
                password="nov@2024",
                database="projects"
            )

        cursor = conn.cursor()
        query = """
        SELECT
            publisher,
            AVG(averageRating) AS avg_rating,
            COUNT(*) AS book_count
        FROM
            book_scape_explorer
        WHERE
            publisher IS NOT NULL AND averageRating IS NOT NULL
        GROUP BY
            publisher
        HAVING
            COUNT(*) > 10
        ORDER BY
            avg_rating DESC
        LIMIT 1;
        """

        cursor.execute(query)
        result = cursor.fetchall()

        if not result:
            st.warning("No publishers with more than 10 books and ratings found.")
            return
        df_result = pd.DataFrame(result, columns=["publisher", "avg_rating", "book_count"])
        st.subheader("Publisher with the Highest Average Rating (More Than 10 Books)")
        st.write(f"The publisher with the highest average rating is **{df_result['publisher'][0]}** with an average rating of **{df_result['avg_rating'][0]:.2f}**.")
        st.write(f"This publisher has published **{df_result['book_count'][0]}** books.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cursor.close()


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
    selected_category = st.selectbox("Choose a category", [100, 200, 300,500])
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
    df = displayrecords()
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
        one()
    elif selected_category == "2.Publisher with the Most Books Published":
        two()
    elif selected_category == "3.Publisher with the Highest Average Rating":
        three()
    elif selected_category == "4.Top 5 Most Expensive Books by Retail Price":
        four()
    elif selected_category == "5.Books Published After 2010 with at Least 500 Pages":
        five()
    elif selected_category == "6.Books with Discounts Greater than 20%":
        six()
    elif selected_category == "7.Average Page Count for eBooks vs Physical Books":
        seven()
    elif selected_category == "8.Top 3 Authors with the Most Books":
        eight()
    elif selected_category == "9.Publishers with More than 10 Books":
        nine()
    elif selected_category == "10.Average Page Count for Each Category":
        ten()
    elif selected_category == "11.Books with More than 3 Authors":
        eleven()
    elif selected_category == "12.Books with Ratings Count Greater Than the Average":
        twelve()
    elif selected_category == "13.Books with the Same Author Published in the Same Year":
        thirteen(df)
    elif selected_category == "14.Books with a Specific Keyword in the Title":
        st.subheader("Search Books by Keyword in Title")

        # Input field for the user to enter the keyword
        keyword = st.text_input("Enter a keyword to search in book titles:")

        if keyword:
            df = fourteen(keyword)
            if not df.empty:
                # Display the results if the DataFrame is not empty
                st.write(f"Books with '{keyword}' in the title:")
                st.dataframe(df)
            else:
                # Notify the user if no results are found
                st.warning(f"No books found with '{keyword}' in the title.")
        else:
            st.info("Please enter a keyword to search.")
    elif selected_category == "15.Year with the Highest Average Book Price":
        fifteen()
    elif selected_category == "16.Count of Authors Who Published 3 Consecutive Years":
        sixteen()
    elif selected_category == "17.authors who have published books in the same year but under different publishers.":
        seventeen()
    elif selected_category == "18.the average amount_retailPrice of eBooks and physical books.":
        eighteen(df)
    elif selected_category == "19.books that have an averageRating that is more than two standard deviations away from the average rating of all books.":
        nineteen()
    elif selected_category == "20.publisher which has the highest average rating among its books, for publishers that have published more than 10 books.":
        twenty()

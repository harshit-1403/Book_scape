# BookScape Explorer

## Project Overview

BookScape Explorer is a powerful web application designed to help users discover, analyze, and gain insights into book data. Leveraging online book APIs and a SQL database, this project provides a seamless platform for exploring book trends, user preferences, and reviews. The application is ideal for avid readers, researchers, and book enthusiasts looking to make informed reading choices and engage with the community.

## Features

- **Fetch Book Data**: Extracts data from the Google Books API to populate a SQL database with comprehensive book information.
- **Data Storage**: Maintains book records in a MySQL database for efficient querying and analysis.
- **Data Analysis**: Offers insights into various aspects of books, such as:
  - Availability of eBooks vs Physical Books
  - Publishers with the Most Books Published
  - Books with Discounts Greater than 20%
  - Top 5 Most Expensive Books by Retail Price
  - And much more (20 categories of insights).
- **Visualization**: Provides visual representations like pie charts and bar graphs for better understanding.
- **Interactive Search**: Enables keyword-based search functionality for book titles.

## Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/) for creating an intuitive user interface.
- **Database**: MySQL for storing and managing book data.
- **APIs**: Google Books API for fetching book details.
- **Data Visualization**: Matplotlib,pandas and Seaborn for generating visual insights.
- **Backend**: Python for handling data processing and analysis.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd bookscape-explorer
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MySQL Database**:
   - Create a database named `projects`.
   - Execute the provided SQL script to set up the required table (`book_scape_explorer`).

4. **Configure API Key**:
   - Replace the `api_key` variable in the code with your Google Books API key.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Database Schema

**Table Name**: `book_scape_explorer`

| Column                  | Data Type   | Description                                     |
|-------------------------|-------------|-------------------------------------------------|
| `book_id`               | VARCHAR     | Unique identifier for the book                 |
| `search_key`            | VARCHAR     | The query used to fetch the book               |
| `book_title`            | VARCHAR     | Title of the book                              |
| `book_subtitle`         | VARCHAR     | Subtitle of the book (if any)                  |
| `book_authors`          | VARCHAR     | Comma-separated list of authors                |
| `book_description`      | TEXT        | Description of the book                        |
| ...                     | ...         | Additional fields as described in the code     |

## Usage

### Fetch Data from API
The application allows users to fetch data by specifying a query. The fetched data is stored in the database and made available for analysis.

### Analyze Data
Use the dropdown menu in the `Data Analysis` section to select from 20 different insights. Examples include:

- Top publishers by book count.
- Distribution of eBooks vs Physical Books.
- Authors who have published consistently for 3 consecutive years.

### Visualize Insights
Charts and graphs generated with Matplotlib and Seaborn make the analysis interactive and easy to understand.



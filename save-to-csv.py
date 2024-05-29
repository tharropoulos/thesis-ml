from dotenv import load_dotenv

load_dotenv()
import os
import csv
import MySQLdb

connection = MySQLdb.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    passwd=os.getenv("DB_PASSWORD"),
    db=os.getenv("DB_NAME"),
    autocommit=True,
    ssl_mode="VERIFY_IDENTITY",
    ssl={"ca": "/etc/ssl/cert.pem"},
)


def export_to_csv(query, file_path):
    cursor = connection.cursor()

    cursor.execute(query)

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description])
        csv_writer.writerows(cursor)

    cursor.close()


queries = {
    "general": """
        SELECT *
        FROM Chat;
    """,
    # number of rewrites, manual_rewrites, revisions, manual_code_intances for each subject
    "total_metrics_by_subject": """
        SELECT subject, 
               SUM(rewrites) AS "Total Copilot Rewrites",
               SUM(manual_rewrites) AS "Total Manual Rewrites",
               SUM(revisions) AS "Total Copilot Revisions",
               SUM(manual_code_intances) AS "Total Manual Code Revisions"
        FROM CopilotFiles
        GROUP BY subject
    """,
    "intervention_ratio_by_subject": """
    SELECT subject,
       SUM(manual_rewrites) / SUM(rewrites) AS "Manual Rewrites to Copilot Changes Ratio",
       SUM(manual_code_intances) / SUM(revisions) AS "Manual Code Revisions to Copilot Revisions Ratio"
    FROM 
       CopilotFiles
    GROUP BY
        subject;
    """,
    "average_metrics_by_subject": """
    SELECT subject,
       AVG(rewrites) AS "Copilot Rewrites",
       AVG(manual_rewrites) AS "Manual Rewrites",
       AVG(revisions) AS "Copilot Revisions",
       AVG(manual_code_intances) AS "Manual Code Revisions"
    FROM 
        CopilotFiles
    GROUP BY 
        subject;
    """,
    # average rating for each subject
    "subject_query": """
    SELECT
        subject,
        AVG(CASE WHEN failed = false AND canceled = false THEN rating ELSE NULL END) AS "Average Rating"
    FROM
        Chat
    GROUP BY
        subject
    ORDER BY
         "Average Rating" DESC
    """,
    # average rating for each type
    "type_query": """
        SELECT
            type,
            AVG(CASE WHEN failed = false AND canceled = false THEN rating ELSE NULL END) AS average_rating
        FROM
            Chat
        GROUP BY
            type
        ORDER BY
            average_rating DESC
    """,
    # total average rating
    "total_query": """
        SELECT 
            AVG(CASE WHEN failed = 0 AND canceled = 0 THEN rating ELSE NULL END) AS total_average_rating
        FROM
            Chat;
    """,
    # percentage of each rating
    "rating_query": """
        SELECT 
            rating,
            COUNT(*) AS "Times Rated",
            (COUNT(*) / (SELECT COUNT(*) FROM Chat WHERE failed = 0 AND canceled = 0)) * 100 AS percentage
        FROM 
            Chat
        WHERE 
            failed = 0 AND canceled = 0
        GROUP BY 
            rating;
    """,
    # total number of failed or canceled chats for each subject and type
    "failing_query": """
        SELECT
            subject,
            SUM(CASE WHEN failed = true OR canceled = true THEN 1 ELSE 0 END) AS num_failed_or_canceled_subject,
            type,
            SUM(CASE WHEN failed = true OR canceled = true THEN 1 ELSE 0 END) AS num_failed_or_canceled_type
        FROM
            Chat
        GROUP BY
            subject, type
    """,
}

current_dir = os.getcwd()

paths = {
    "general": os.path.join(current_dir, "export", "general.csv"),
    "total_metrics_by_subject": os.path.join(
        current_dir, "export", "total_metrics_by_subject.csv"
    ),
    "intervention_ratio_by_subject": os.path.join(
        current_dir, "export", "intervention_ratio_by_subject.csv"
    ),
    "average_metrics_by_subject": os.path.join(
        current_dir, "export", "average_metrics_by_subject.csv"
    ),
    "subject_query": os.path.join(current_dir, "export", "subject.csv"),
    "type_query": os.path.join(current_dir, "export", "type.csv"),
    "failing_query": os.path.join(current_dir, "export", "fails.csv"),
    "total_query": os.path.join(current_dir, "export", "total.csv"),
    "rating_query": os.path.join(current_dir, "export", "rating.csv"),
}

for key, value in paths.items():
    if not os.path.exists(os.path.dirname(value)):
        os.makedirs(os.path.dirname(value))

for key, value in queries.items():
    export_to_csv(value, paths[key])

connection.close()

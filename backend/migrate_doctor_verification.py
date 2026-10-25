import sqlite3
from pathlib import Path

from app.database.session import Base, engine


# ---------------------------------------------------------
# DATABASE LOCATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"


def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    return any(column[1] == column_name for column in columns)


def run_migration():
    print("Starting doctor verification database migration...")

    # -----------------------------------------------------
    # Check database exists
    # -----------------------------------------------------
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Make sure backend/app.db exists before running this.")
        return

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    try:
        # -------------------------------------------------
        # Add doctor_id column if it does not exist
        # -------------------------------------------------
        if not column_exists(cursor, "users", "doctor_id"):
            print("Adding doctor_id column...")
            cursor.execute(
                """
                ALTER TABLE users
                ADD COLUMN doctor_id VARCHAR
                """
            )
        else:
            print("doctor_id column already exists.")

        # -------------------------------------------------
        # Add verification_status column if it does not exist
        # -------------------------------------------------
        if not column_exists(
            cursor,
            "users",
            "verification_status",
        ):
            print("Adding verification_status column...")
            cursor.execute(
                """
                ALTER TABLE users
                ADD COLUMN verification_status VARCHAR
                """
            )
        else:
            print("verification_status column already exists.")

        # -------------------------------------------------
        # Set existing users to approved temporarily
        # -------------------------------------------------
        #
        # This avoids locking existing accounts out during
        # the migration. New doctor accounts will later use
        # "pending" through the registration code.
        #
        cursor.execute(
            """
            UPDATE users
            SET verification_status = 'approved'
            WHERE verification_status IS NULL
            """
        )

        # -------------------------------------------------
        # Make the current admin approved
        # -------------------------------------------------
        cursor.execute(
            """
            UPDATE users
            SET verification_status = 'approved'
            WHERE role = 'admin'
            """
        )

        connection.commit()

        print("User table migration completed.")

    except Exception as e:
        connection.rollback()
        print("Migration failed.")
        print(f"Error: {e}")
        raise

    finally:
        connection.close()

    # -----------------------------------------------------
    # Create new tables from SQLAlchemy models
    # -----------------------------------------------------
    print("Creating missing tables...")

    Base.metadata.create_all(bind=engine)

    print("Database migration completed successfully.")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    run_migration()
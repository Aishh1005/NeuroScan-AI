import sqlite3


def main():
    print("\n=== Add Demo Doctor ===\n")

    doctor_name = input("Doctor name: ").strip()
    doctor_id = input("Doctor ID: ").strip()
    verification_code = input("Verification code: ").strip().upper()

    if not doctor_name or not doctor_id or not verification_code:
        print("\nError: All fields are required.")
        return

    connection = sqlite3.connect("app.db")
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO demo_doctors
            (doctor_id, name, verification_code, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                doctor_id,
                doctor_name,
                verification_code,
                "verified",
            ),
        )

        connection.commit()

        print("\nDemo doctor added successfully!")
        print(f"Name: {doctor_name}")
        print(f"Doctor ID: {doctor_id}")
        print(f"Verification Code: {verification_code}")

    except sqlite3.IntegrityError:
        print("\nError: This Doctor ID already exists.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
import os
import shutil
import hashlib

# ============================================================
# 1. PATHS
# ============================================================

# YOUR CURRENT 7,200-IMAGE DATASET
CURRENT_DATASET = r"C:\Users\Aishwarya Patil\Desktop\tumor copy 1\brain-tumor-project-backup 2\dataset"

# YOUR NEW 3,264-IMAGE DATASET
NEW_DATASET = r"C:\Users\Aishwarya Patil\Downloads\Brain Tumor Classification MRI Dataset (Glioma, Me\Brain Tumor Classification MRI Dataset (Glioma, Me"

# NEW SEPARATE COMBINED DATASET
COMBINED_DATASET = os.path.join(
    os.path.dirname(CURRENT_DATASET),
    "combined_dataset"
)


# ============================================================
# 2. FOLDERS
# ============================================================

CURRENT_TRAIN = os.path.join(CURRENT_DATASET, "Training")
CURRENT_TEST = os.path.join(CURRENT_DATASET, "Testing")

NEW_TRAIN = os.path.join(NEW_DATASET, "Train")
NEW_TEST = os.path.join(NEW_DATASET, "Test")

COMBINED_TRAIN = os.path.join(COMBINED_DATASET, "Training")
COMBINED_TEST = os.path.join(COMBINED_DATASET, "Testing")


# ============================================================
# 3. CLASS MAPPING
# ============================================================

# New dataset              → Current/project class

CLASS_MAPPING = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "pituitary_tumor": "pituitary",
    "no_tumor": "notumor"
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# 4. GET IMAGES
# ============================================================

def get_images(folder):

    images = []

    if not os.path.exists(folder):
        print("\nERROR: Folder does not exist:")
        print(folder)
        return images

    for root, dirs, files in os.walk(folder):

        for file in files:

            extension = os.path.splitext(file)[1].lower()

            if extension in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, file))

    return images


# ============================================================
# 5. SHA-256 HASH
# ============================================================

def get_hash(filepath):

    sha256 = hashlib.sha256()

    try:

        with open(filepath, "rb") as f:

            while True:

                data = f.read(1024 * 1024)

                if not data:
                    break

                sha256.update(data)

        return sha256.hexdigest()

    except Exception as e:

        print("\nCould not read:")
        print(filepath)
        print("Error:", e)

        return None


# ============================================================
# 6. BUILD HASH DATABASE
# ============================================================

def build_hash_database(folder):

    hash_database = {}

    images = get_images(folder)

    print("\nFolder:")
    print(folder)

    print("Images:", len(images))

    for index, image in enumerate(images, start=1):

        file_hash = get_hash(image)

        if file_hash:

            # Store first occurrence
            if file_hash not in hash_database:
                hash_database[file_hash] = image

        if index % 500 == 0:
            print(f"Processed {index}/{len(images)}")

    return hash_database


# ============================================================
# 7. CREATE COMBINED FOLDERS
# ============================================================

def create_combined_folders():

    classes = [
        "glioma",
        "meningioma",
        "pituitary",
        "notumor"
    ]

    for class_name in classes:

        os.makedirs(
            os.path.join(COMBINED_TRAIN, class_name),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(COMBINED_TEST, class_name),
            exist_ok=True
        )


# ============================================================
# 8. COPY CURRENT DATASET
# ============================================================

def copy_current_dataset():

    print("\n" + "=" * 70)
    print("COPYING CURRENT DATASET")
    print("=" * 70)

    # -----------------------------
    # CURRENT TRAINING
    # -----------------------------

    for class_name in os.listdir(CURRENT_TRAIN):

        source_folder = os.path.join(
            CURRENT_TRAIN,
            class_name
        )

        if not os.path.isdir(source_folder):
            continue

        destination_folder = os.path.join(
            COMBINED_TRAIN,
            class_name
        )

        images = get_images(source_folder)

        for image in images:

            destination = os.path.join(
                destination_folder,
                os.path.basename(image)
            )

            shutil.copy2(image, destination)

    # -----------------------------
    # CURRENT TESTING
    # -----------------------------

    for class_name in os.listdir(CURRENT_TEST):

        source_folder = os.path.join(
            CURRENT_TEST,
            class_name
        )

        if not os.path.isdir(source_folder):
            continue

        destination_folder = os.path.join(
            COMBINED_TEST,
            class_name
        )

        images = get_images(source_folder)

        for image in images:

            destination = os.path.join(
                destination_folder,
                os.path.basename(image)
            )

            shutil.copy2(image, destination)

    print("\nCurrent dataset copied.")
    print("Original dataset was NOT changed.")


# ============================================================
# 9. GET SAFE NEW IMAGES
# ============================================================

def find_safe_new_images(new_folder, current_hashes):

    safe_images = []
    duplicate_images = []

    images = get_images(new_folder)

    for image in images:

        file_hash = get_hash(image)

        if not file_hash:
            continue

        # Already exists in CURRENT dataset
        if file_hash in current_hashes:

            duplicate_images.append(image)

        else:

            safe_images.append(
                (image, file_hash)
            )

    return safe_images, duplicate_images


# ============================================================
# 10. REMOVE DUPLICATES BETWEEN NEW TRAIN AND NEW TEST
# ============================================================

def remove_internal_duplicates(
    safe_train,
    safe_test
):

    train_hashes = {
        file_hash
        for image, file_hash in safe_train
    }

    final_test = []
    internal_duplicates = []

    for image, file_hash in safe_test:

        if file_hash in train_hashes:

            internal_duplicates.append(image)

        else:

            final_test.append(
                (image, file_hash)
            )

    return safe_train, final_test, internal_duplicates


# ============================================================
# 11. COPY SAFE NEW TRAINING
# ============================================================

def copy_safe_training(safe_images):

    added = 0

    for image, file_hash in safe_images:

        new_class = os.path.basename(
            os.path.dirname(image)
        )

        if new_class not in CLASS_MAPPING:
            continue

        combined_class = CLASS_MAPPING[new_class]

        destination_folder = os.path.join(
            COMBINED_TRAIN,
            combined_class
        )

        filename = "new_" + os.path.basename(image)

        destination = os.path.join(
            destination_folder,
            filename
        )

        counter = 1

        while os.path.exists(destination):

            base, ext = os.path.splitext(filename)

            destination = os.path.join(
                destination_folder,
                f"{base}_{counter}{ext}"
            )

            counter += 1

        shutil.copy2(
            image,
            destination
        )

        added += 1

    return added


# ============================================================
# 12. COPY SAFE NEW TESTING
# ============================================================

def copy_safe_testing(safe_images):

    added = 0

    for image, file_hash in safe_images:

        new_class = os.path.basename(
            os.path.dirname(image)
        )

        if new_class not in CLASS_MAPPING:
            continue

        combined_class = CLASS_MAPPING[new_class]

        destination_folder = os.path.join(
            COMBINED_TEST,
            combined_class
        )

        filename = "new_" + os.path.basename(image)

        destination = os.path.join(
            destination_folder,
            filename
        )

        counter = 1

        while os.path.exists(destination):

            base, ext = os.path.splitext(filename)

            destination = os.path.join(
                destination_folder,
                f"{base}_{counter}{ext}"
            )

            counter += 1

        shutil.copy2(
            image,
            destination
        )

        added += 1

    return added


# ============================================================
# 13. COUNT DATASET
# ============================================================

def count_dataset(folder):

    classes = [
        "glioma",
        "meningioma",
        "pituitary",
        "notumor"
    ]

    counts = {}

    total = 0

    for class_name in classes:

        class_folder = os.path.join(
            folder,
            class_name
        )

        count = len(
            get_images(class_folder)
        )

        counts[class_name] = count
        total += count

    return counts, total


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAFE COMBINED DATASET CREATION")
    print("=" * 70)

    print("\nIMPORTANT:")
    print("Original datasets will NOT be deleted.")
    print("Original datasets will NOT be moved.")
    print("Original datasets will NOT be modified.")
    print("Only COPY operations will be performed.")

    # --------------------------------------------------------
    # Create combined folders
    # --------------------------------------------------------

    create_combined_folders()

    # --------------------------------------------------------
    # Build current dataset hash database
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING CURRENT TRAINING")
    print("=" * 70)

    current_train_hashes = build_hash_database(
        CURRENT_TRAIN
    )

    print("\n" + "=" * 70)
    print("CHECKING CURRENT TESTING")
    print("=" * 70)

    current_test_hashes = build_hash_database(
        CURRENT_TEST
    )

    # Combine both current hashes.
    # This protects against:
    # Current Train ↔ New Test
    # Current Test ↔ New Train

    current_hashes = set(
        current_train_hashes.keys()
    ) | set(
        current_test_hashes.keys()
    )

    print("\nUnique hashes in current dataset:")
    print(len(current_hashes))

    # --------------------------------------------------------
    # Copy current dataset
    # --------------------------------------------------------

    copy_current_dataset()

    # --------------------------------------------------------
    # Find safe NEW TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING NEW TRAINING")
    print("=" * 70)

    safe_train, duplicate_train = find_safe_new_images(
        NEW_TRAIN,
        current_hashes
    )

    print("\nNew training images:")
    print(len(get_images(NEW_TRAIN)))

    print("Exact duplicates:")
    print(len(duplicate_train))

    print("Safe new training images:")
    print(len(safe_train))

    # --------------------------------------------------------
    # Find safe NEW TESTING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING NEW TESTING")
    print("=" * 70)

    safe_test, duplicate_test = find_safe_new_images(
        NEW_TEST,
        current_hashes
    )

    print("\nNew testing images:")
    print(len(get_images(NEW_TEST)))

    print("Exact duplicates:")
    print(len(duplicate_test))

    print("Safe new testing images:")
    print(len(safe_test))

    # --------------------------------------------------------
    # Protect against new train/test duplicates
    # --------------------------------------------------------

    (
        safe_train,
        safe_test,
        internal_duplicates
    ) = remove_internal_duplicates(
        safe_train,
        safe_test
    )

    print("\nNew Training ↔ New Testing exact duplicates:")
    print(len(internal_duplicates))

    # --------------------------------------------------------
    # Copy safe new training
    # --------------------------------------------------------

    added_train = copy_safe_training(
        safe_train
    )

    # --------------------------------------------------------
    # Copy safe new testing
    # --------------------------------------------------------

    added_test = copy_safe_testing(
        safe_test
    )

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    train_counts, train_total = count_dataset(
        COMBINED_TRAIN
    )

    test_counts, test_total = count_dataset(
        COMBINED_TEST
    )

    print("\n")
    print("=" * 70)
    print("FINAL COMBINED DATASET")
    print("=" * 70)

    print("\nTRAINING:")

    for class_name, count in train_counts.items():
        print(f"{class_name:12s}: {count}")

    print(f"\nTOTAL TRAINING: {train_total}")

    print("\nTESTING:")

    for class_name, count in test_counts.items():
        print(f"{class_name:12s}: {count}")

    print(f"\nTOTAL TESTING: {test_total}")

    print(f"\nTOTAL IMAGES: {train_total + test_total}")

    print("\n" + "=" * 70)
    print("NEW IMAGES ADDED")
    print("=" * 70)

    print("Training:", added_train)
    print("Testing :", added_test)

    print("\nCombined dataset location:")
    print(COMBINED_DATASET)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nIMPORTANT:")
    print("Your original 7,200-image dataset was NOT modified.")
    print("Your downloaded 3,264-image dataset was NOT modified.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
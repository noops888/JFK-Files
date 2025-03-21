import os
import glob

# Get all PDF files in original_files
pdf_files = glob.glob('./original_files/*.pdf')
pdf_basenames = [os.path.splitext(os.path.basename(f))[0] for f in pdf_files]

# Get all TXT files in extracted_text
txt_files = glob.glob('./extracted_text/*.txt')
txt_basenames = [os.path.splitext(os.path.basename(f))[0] for f in txt_files]

# Find PDFs without corresponding TXT files
missing_txt = set(pdf_basenames) - set(txt_basenames)

# Print results
print(f"Total PDFs: {len(pdf_basenames)}")
print(f"Total TXTs: {len(txt_basenames)}")
print(f"Missing TXT files: {len(missing_txt)}")

if missing_txt:
    print("\nMissing files:")
    for filename in sorted(missing_txt):
        print(f"{filename}.pdf")

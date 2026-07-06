# CryoSPARC_Live_Report
Generates a PDF report of a CryoSPARC Live session. I have no affiliation with Structura and plots are re-generated from the underlying data, so they may vary slightly from what you see in the CryoSPARC web viewer. 

## Installation

This script requires that you have conda installed, and that you have write and execute privileges in your file system. If you do not have write and/or execute privileges, speak with your IT folks and they may be able to help you with specific exceptions and/or install this for you. 

1. Download the latest version of the code
```bash
git clone https://github.com/tlevitz/CryoSPARC_Live_Report.git /path/to/your/scripts/folder
```

2. Create a live_report conda environment
```bash
conda env create -f environment.yml
```

3. Activate the environment
```bash
conda activate live_report
```

4. Run the script
```bash
python /path/to/generate_live_report.py /path/to/CS-folder
```
Advanced options:
```bash
usage: generate_live_report.py [-h] [--session SESSION] [--refine-job REFINE_JOB]
                               project_dir

Generate a CryoSPARC Live session PDF report.

positional arguments:
  project_dir           Path to CryoSPARC project directory

optional arguments:
  -h, --help            show this help message and exit
  --session SESSION     Session dir/uid (default: S1)
  --refine-job REFINE_JOB
                        Refinement job uid (default: auto-detected from session)
```

_Notes_
1. If no 2D classification, ab-initio, or homogeneous refinement jobs were run during the Live session, those segments will automatically be omitted from the report
2. If the raw data have been moved or deleted since the session was created, or if metadata information (.xml or .mdoc) files are not in the same folder as the micrographs, the positional information plots will not be able to be generated and will automatically be omitted from the report. The script can resolve symlinks, so as long as the true files are in their original directory structure as written by SerialEM or EPU, it should be able to find the positional information.
3. These files are meant to be viewed on a computer/tablet with zoom, and some plots may be too small for printing. The final resolution of the PDF is a balance between readability, speed, size, and quality, and should zoom sufficiently for quality assessment purposes.
4. These scripts were generated with the assistance of GPT4DFCI, a private, HIPAA-secure endpoint to GPT-4o provided by DFCI

## Example Report
<img width="678" height="876" alt="image" src="https://github.com/user-attachments/assets/32826bbf-16b3-4c6c-887e-ff7e1322a58d" />
<img width="675" height="872" alt="image" src="https://github.com/user-attachments/assets/f7e7d9b8-eec8-403a-ba16-5aa469c903bf" />
<img width="677" height="876" alt="image" src="https://github.com/user-attachments/assets/9875d375-7141-458f-b885-1af42a3f2600" />
<img width="676" height="877" alt="image" src="https://github.com/user-attachments/assets/84540aeb-ee3b-480a-ba07-30fe5372f72c" />
<img width="673" height="865" alt="image" src="https://github.com/user-attachments/assets/eb9edbbe-d2b2-4be8-98bf-4d72a71579df" />
<img width="675" height="878" alt="image" src="https://github.com/user-attachments/assets/0fa97314-a04a-411b-b618-72d018cf3f3d" />
<img width="675" height="868" alt="image" src="https://github.com/user-attachments/assets/d9d28e0d-7c8e-4b58-8f6c-bd1e9a8eec4a" />
<img width="676" height="876" alt="image" src="https://github.com/user-attachments/assets/2d8b35c8-0a42-47bb-a501-67def496e05e" />
<img width="677" height="876" alt="image" src="https://github.com/user-attachments/assets/a80a0bd0-c963-4ebe-996c-6e65d7e12974" />
<img width="678" height="871" alt="image" src="https://github.com/user-attachments/assets/ea427d70-1b06-4a5e-a879-ce54e85baee1" />
<img width="676" height="873" alt="image" src="https://github.com/user-attachments/assets/029e82bb-31c7-4f95-9034-444ac7e21db6" />

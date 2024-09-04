Global database on school-age digital connectivity ReadMe


Overview

This README provides insights into the "Global School-Age Digital Connectivity" project. The project utilizes a dataset describing the percentage of school-age children with internet access at home across various countries, categorized by different demographics and economic indicators.

Project Files

Global School-Age Digital Connectivity.ipynb: Jupyter notebook containing the data analysis code.
Project Report.pdf: Comprehensive report detailing the data preparation, error handling, data exploration, and findings.
Setup Instructions
File Organization: Ensure all project files are kept in the same directory to facilitate smooth execution of scripts and easy access to data.


Running the Notebook:

Open a terminal or command prompt.
Navigate to the directory containing the project files.
Run the command jupyter notebook to start the Jupyter Notebook server.
Open the notebook file assignment_1_vivek_4015465.ipynb from the Jupyter dashboard to view and run the analysis.



Data Preparation and Analysis

Data Cleaning: The dataset was cleansed of null values, incorrect data entries, and redundant columns to ensure data quality and usability for analysis.

Data Transformation: Columns were reformatted, and data types were converted to facilitate analysis. Multivalued attributes were split and structured for relational integration.

Error Handling: Various data-related errors were identified and rectified, including type conversions and reference errors.



Indicator definition	

* Definition: School-age digital connectivity data set – Percentage of children in a school attendance age (approximately 3-17 years old depending on the country) that have internet connection at home

Methodology	
* Unit of measure: Percentage
* Time frame for survey: Household survey data as of year 2010 onwards are used to calculate the indicator. For countries with multiple years of data, the most recent dataset is used.

Glossary - the database contains the following	
* ISO: Three-digit alphabetical codes International Standard ISO 3166-1 assigned by the International Organization for Standardization (ISO). The latest version is available online at http://www.iso.org/iso/home/standards/country_codes.htm. (column A)
* Countries and areas: The UNICEF Global databases contain a set of 202 countries as reported on through the State of the World's Children Statistical Annex 2017 (column B)
	
* Data Source: Short name for data source, followed by the year(s) in which the data collection (e.g., survey interviews) took place (column K)
* Time period: Represents the year(s) in which the data collection (e.g. survey interviews) took place. (column L)
	
* Region, Sub-region: UNICEF regions (column C) and UNICEF Sub-regions (column D)
EAP	East Asia and the Pacific
ECA	Europe and Central Asia
EECA	Eastern Europe and Central Asia
ESA	Eastern and Southern Africa
LAC	Latin America and the Caribbean
MENA	Middle East and North Africa
NA	North America
SA	South Asia
SSA	Sub-Saharan Africa
WCA	West and Central Africa

* Development regions: Economies are currently divided into four income groupings: low, lower-middle, upper-middle, and high. Income is measured using gross national income (GNI) per capita, in U.S. dollars, converted from local currency using the World Bank Atlas method (column E).

* Total: The overall percentage of children in a school attendance age that have internet connection at home

* Rural (Residence): The percentage of children (live in rural area) in a school attendance age that have internet connection at home
	
* Urban (Residence): The percentage of children (live in urban area) in a school attendance age that have internet connection at home
(For Residence, there are only two values: Rural and Urban)

* Poorest (Wealth quintile): The percentage of children (from the poorest according to wealth quintile) in a school attendance age that have internet connection at home

* Richest (Wealth quintile): The percentage of children (from the poorest according to wealth quintile) in a school attendance age that have internet connection at home


Data Files: 
* Total School Age: The data about all children in a school attendance age (approximately 3-17 years old depending on the country) 
* Primary: The data about children in a primary school
* Secondary: The data about children in a Secondary school.
	
Disclaimer	
All reasonable precautions have been taken to verify the information in this database. In no event shall UNICEF be liable for damages arising from its use or interpretation	
Week 3 – Advanced Data Analysis and Visualization in Logistics

1. Project Overview

This project demonstrates advanced data analysis and visualization techniques using Python to understand logistics performance, identify operational challenges, and support data-driven decision-making.

The analysis uses 54 simulated cleaned logistics records carried forward from the Week 2 project.

2. Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Calculate logistics Key Performance Indicators (KPIs).
- Analyze delivery performance and freight charges.
- Identify trends and patterns using visualizations.
- Apply regression to predict delivery times.
- Use K-Means clustering to identify delivery groups.
- Explain findings, limitations, and business implications.

3. Dataset Information

The project uses 54 simulated logistics orders.

The Brazilian E-Commerce Public Dataset by Olist is used as a real-world research reference.

Dataset reference: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Note: The project's numerical results are based on simulated data, not actual Olist records.

4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Data Visualization
- Regression Analysis
- K-Means Clustering

5. Key Performance Indicators

KPI| Result
Total cleaned orders| 54
Delivered orders| 46
Shipped orders| 5
Canceled orders| 3
On-time delivery rate| 50.0%
Average delivery time| 11.91 days
Average freight charge| R$ 157.61

The on-time delivery rate is calculated using delivered orders only.

6. Advanced Analysis

Exploratory Data Analysis

Analyzed delivery durations, order outcomes, freight charges, and regional performance.

Regression Analysis

A regression model was used to demonstrate delivery-time prediction.

- Baseline MAE: 6.143 days
- Ridge Regression MAE: 4.925 days
- R² Score: -0.582

Although the regression model improved MAE compared with the baseline, its negative R² indicates limited predictive reliability.

Clustering Analysis

K-Means clustering was applied to identify similar groups of logistics orders.

- Number of clusters: 3
- Silhouette score: 0.486

These results are descriptive and require validation using larger real-world datasets.

7. Data Visualizations

The project includes eight charts covering:

- Order outcomes
- Delivery-time distribution
- Regional on-time performance
- Freight sensitivity
- Regression model comparison
- Predicted versus actual delivery time
- Delivery clusters
- Distance and freight relationships

8. Project Files

- "week3_analysis.py" – Main Python analysis program.
- "requirements.txt" – Required Python libraries.
- "data/" – Cleaned logistics dataset.
- "outputs/" – Statistical summaries, model predictions, and clustering results.
- "figures/" – Eight analytical charts.
- "Week_3_Advanced_Logistics_Analytics_Report.docx" – Detailed internship report.

9. How to Run the Project

Install the dependencies:

"python -m pip install -r requirements.txt"

Execute the program:

"python week3_analysis.py"

Review the generated results in the "outputs/" and "figures/" folders.

10. Business Insights and Reflection

Data analytics can help logistics managers investigate delivery delays, identify operational differences across regions, and monitor transportation-related charges.

However, the reliability of these insights depends on the quality and representativeness of the available data.

The findings in this project are educational demonstrations and should not be interpreted as verified real-world logistics performance or financial savings.

11. Conclusion

The project demonstrates how Python-based data analysis, predictive modeling, clustering, and visualization can support logistics performance evaluation.

It also highlights the importance of validating analytical results and understanding model limitations before using them for operational decision-making.

Data Analytics Internship – Week 3 Project

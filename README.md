# University Admission Analysis by Joshua Osarumwense

This project focuses on analysing the factors that influence university admission, including **GRE scores**, **TOEFL scores**, **CGPA**, **SOP**, and **Research**. The goal is to explore correlations between these variables and predict the **Chance of Admission**. This analysis was performed using **Python**, **Pandas**, **Seaborn**, **Scikit-learn**, and **Matplotlib**.
## 🧑🏽‍💻 Technologies Used

- **Python**: Core programming language for data analysis.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib** & **Seaborn**: For data visualization and plotting.
- **Scikit-learn**: For machine learning and model evaluation.
- **StandardScaler**: Used for feature scaling before training machine learning models.
- **GridSearchCV**: Used to tune the linear regression model's hyperparameters.

## 🔍 Key Features

- **Data Cleaning**: Handled missing values, removed duplicates, and cleaned column names.
- **Exploratory Data Analysis (EDA)**: Performed initial data exploration, including statistical summaries and visualizations (histograms, pie charts, heatmaps).
- **Feature Engineering**: Applied scaling to ensure that features are appropriately scaled for the model.
- **Correlation Analysis**: Investigated how various factors, such as GRE score, TOEFL score, and CGPA, correlate with the chance of admission.
- **Modeling**: Built and evaluated a **Linear Regression** model to predict the "Chance of Admission" based on various factors. Tuned the model using **GridSearchCV**.
- **Visualization**: Visualized the data distributions and relationships using various plots, including **histograms**, **pie charts**, and **scatter plots**.


## 📊 Data Visualization

### Distribution of GRE Scores
The **distribution of GRE scores** shows a relatively normal distribution with the highest frequency around the 320-325 score range. This suggests that most students tend to score within this range, which aligns with competitive university admissions.

![Distribution of GRE Scores](https://github.com/Josh-Osas/University-Admission-Data-Analysis/blob/main/GRE%20Histogram.png)

### Distribution of TOEFL Scores
The **TOEFL score distribution** is somewhat uniform, with the highest frequency around the 110-115 range, indicating that a large number of students perform well in the TOEFL exam, which is often a prerequisite for admission to international universities.

![Distribution of TOEFL Scores](https://github.com/Josh-Osas/University-Admission-Data-Analysis/blob/main/TOEFL%20Histogram.png)

### Research Experience
A **pie chart** representing the distribution of students with and without research experience indicates that **44%** of the applicants have research experience. This is significant, as research experience can be a deciding factor for many graduate programs.

![Research Experience](https://github.com/Josh-Osas/University-Admission-Data-Analysis/blob/main/Research%20Pie%20Chart.png)

### Correlation Heatmap
The **correlation heatmap** provides a detailed analysis of the relationships between various factors and the **Chance of Admission**. Here are the key insights:

![Correlation Heatmap](https://github.com/Josh-Osas/University-Admission-Data-Analysis/blob/main/Correlation%20Heatmap.png)

## 🔍 Insights from the Heatmap

- **GRE Score** and **TOEFL Score**:
  - There is a **strong positive correlation** (0.83) between **GRE scores** and **TOEFL scores**. This suggests that students who perform well on the GRE also tend to perform well on the TOEFL exam, which is typical for applicants aiming for competitive universities.
  
- **CGPA** and **Chance of Admission**:
  - **CGPA** has the **strongest positive correlation** with the **Chance of Admission** (0.88). This indicates that students with higher CGPA scores are more likely to be admitted. It highlights the importance of academic performance in the admission process.

- **Research Experience**:
  - The **Research** variable shows a moderate positive correlation (0.55) with the **Chance of Admission**. While not as strong as CGPA, students with research experience still have a higher chance of admission. This suggests that research experience, while important, may not be as crucial as academic performance (CGPA) or standardized test scores (GRE and TOEFL).

- **SOP** (Statement of Purpose) and **LOR** (Letters of Recommendation):
  - Both **SOP** and **LOR** have moderate correlations with **Chance of Admission** (0.68 and 0.65, respectively). This reinforces the idea that a well-crafted SOP and strong recommendation letters play a crucial role in the admission process.
  
- **University Rating**:
  - **University Rating** has a moderate positive correlation (0.69) with **Chance of Admission**, suggesting that students applying to higher-rated universities tend to have a higher chance of admission. This may reflect the selectivity of top-tier universities.

## 💡 Recommendations

Based on the analysis, the following recommendations are suggested for improving the chances of admission:

1. **Improve GRE and TOEFL Scores**: Since these scores have strong correlations with the chance of admission, applicants should aim for higher scores on these exams to increase their chances.
2. **Focus on Academic Performance**: **CGPA** is the strongest predictor of admission. Students should maintain a strong academic record to improve their admission prospects.
3. **Enhance Research Experience**: Students with research experience have a higher chance of admission, especially for graduate programs. Applicants should consider gaining research experience where possible.
4. **Strong SOP and LOR**: Crafting a strong **Statement of Purpose (SOP)** and obtaining solid **Letters of Recommendation (LOR)** can positively impact admission chances, as these factors have a moderate correlation with admission.
5. **Target High-Rated Universities**: Applicants applying to higher-rated universities tend to have higher chances of admission. However, this also means a more competitive selection process, so applicants should ensure they meet all requirements.







import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telecom Customer Churn Prediction")
st.markdown(
    """
    Predict customer churn using Machine Learning and understand
    **why a customer is at risk** and **what actions can be taken
    to improve retention**.
    """
)

# ============================================================
# LOAD DATA
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "data" / "telecom_churn.csv")
    return data
df = load_data()

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins"
]

TARGET = "Churn"

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


model, X_train, X_test, y_train, y_test = train_model(X, y)

# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Customer Prediction",
        "📊 Feature Importance",
        "📈 Model Performance",
        "📋 Dataset Overview"
    ]
)

# ============================================================
# PAGE 1 - CUSTOMER PREDICTION
# ============================================================

if page == "🔮 Customer Prediction":

    st.header("🔮 Predict Customer Churn")

    st.write(
        "Enter the customer's information below. "
        "The model will estimate the probability that the customer will churn."
    )

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        account_weeks = st.number_input(
            "Account Weeks",
            min_value=1,
            max_value=300,
            value=100
        )

        contract_renewal = st.selectbox(
            "Contract Renewal",
            options=[1, 0],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        data_plan = st.selectbox(
            "Data Plan",
            options=[1, 0],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        data_usage = st.number_input(
            "Data Usage",
            min_value=0.0,
            max_value=10.0,
            value=0.8,
            step=0.1
        )

    with col2:

        cust_serv_calls = st.number_input(
            "Customer Service Calls",
            min_value=0,
            max_value=20,
            value=1
        )

        day_mins = st.number_input(
            "Day Minutes",
            min_value=0.0,
            max_value=500.0,
            value=180.0,
            step=1.0
        )

        day_calls = st.number_input(
            "Day Calls",
            min_value=0,
            max_value=300,
            value=100
        )

    with col3:

        monthly_charge = st.number_input(
            "Monthly Charge",
            min_value=0.0,
            max_value=200.0,
            value=55.0,
            step=0.5
        )

        overage_fee = st.number_input(
            "Overage Fee",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )

        roam_mins = st.number_input(
            "Roaming Minutes",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    customer = pd.DataFrame(
        [[
            account_weeks,
            contract_renewal,
            data_plan,
            data_usage,
            cust_serv_calls,
            day_mins,
            day_calls,
            monthly_charge,
            overage_fee,
            roam_mins
        ]],
        columns=FEATURES
    )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(customer)[0][1]

        probability_percent = probability * 100

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability < 0.30:
            risk_level = "Low Risk"
        elif probability < 0.60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            if prediction == 1:
                st.error("⚠️ Customer likely to churn")
            else:
                st.success("✅ Customer likely to stay")

        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )

        with result_col3:

            st.metric(
                "Risk Level",
                risk_level
            )

        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.progress(
            min(int(probability_percent), 100)
        )

        # ====================================================
        # CUSTOMER-SPECIFIC INSIGHTS
        # ====================================================

        st.divider()

        st.subheader("🔍 Customer Risk Insights")

        insights = []

        # Contract
        if contract_renewal == 0:
            insights.append(
                (
                    "🔴",
                    "Contract Renewal",
                    "The customer has not renewed the contract.",
                    "Review the customer's contract status and proactively discuss renewal options."
                )
            )

        # Customer service calls
        if cust_serv_calls >= 3:
            insights.append(
                (
                    "🔴",
                    "Customer Service Calls",
                    f"The customer has made {cust_serv_calls} service calls.",
                    "Contact the customer and investigate unresolved service issues."
                )
            )

        elif cust_serv_calls == 2:
            insights.append(
                (
                    "🟠",
                    "Customer Service Calls",
                    "The customer has made multiple service calls.",
                    "Check whether the customer's previous issues were properly resolved."
                )
            )

        # Day minutes
        churn_day_mins = df[df["Churn"] == 1]["DayMins"].mean()

        if day_mins > churn_day_mins:
            insights.append(
                (
                    "🟠",
                    "High Usage",
                    f"Day usage is above the average observed among churned customers "
                    f"({churn_day_mins:.1f} minutes).",
                    "Review whether the customer's current plan is suitable for their usage."
                )
            )

        # Overage
        churn_overage = df[df["Churn"] == 1]["OverageFee"].mean()

        if overage_fee > churn_overage:
            insights.append(
                (
                    "🟠",
                    "Overage Fee",
                    f"Overage fee is above the average observed among churned customers "
                    f"(${churn_overage:.2f}).",
                    "Review the customer's plan and investigate whether a better allowance could reduce overage charges."
                )
            )

        # Monthly charge
        churn_charge = df[df["Churn"] == 1]["MonthlyCharge"].mean()

        if monthly_charge > churn_charge:
            insights.append(
                (
                    "🟡",
                    "Monthly Charge",
                    f"Monthly charge is above the average observed among churned customers "
                    f"(${churn_charge:.2f}).",
                    "Check whether the customer's plan provides sufficient value for the amount they pay."
                )
            )

        # Data plan
        if data_plan == 0:
            insights.append(
                (
                    "🟡",
                    "Data Plan",
                    "The customer does not currently have a data plan.",
                    "Evaluate whether a suitable data plan could better match the customer's usage."
                )
            )

        # Day minutes very high
        if day_mins > df["DayMins"].quantile(0.75):
            insights.append(
                (
                    "🟠",
                    "High Day Usage",
                    "The customer's day-time usage is in the higher range of the dataset.",
                    "Consider whether the customer needs a plan with a higher usage allowance."
                )
            )

        # ----------------------------------------------------
        # DISPLAY INSIGHTS
        # ----------------------------------------------------

        if len(insights) == 0:

            st.success(
                "No major rule-based risk signals were identified "
                "from the customer's entered information."
            )

        else:

            for icon, title, observation, action in insights:

                with st.container(border=True):

                    st.markdown(
                        f"### {icon} {title}"
                    )

                    st.write(
                        f"**Observation:** {observation}"
                    )

                    st.write(
                        f"**Recommended Action:** {action}"
                    )

        # ====================================================
        # RETENTION ACTION PLAN
        # ====================================================

        st.divider()

        st.subheader("🛠 Recommended Retention Action Plan")

        if prediction == 1:

            st.warning(
                "This customer has been classified as a churn-risk customer. "
                "The following actions can be considered."
            )

            actions = [
                "📞 Contact the customer proactively.",
                "🔧 Investigate and resolve repeated customer-service issues.",
                "📄 Review contract renewal status.",
                "💰 Review monthly charges and overage fees.",
                "📦 Check whether the current plan matches the customer's usage.",
                "🎁 Consider a targeted retention offer where appropriate.",
                "📊 Monitor the customer after intervention."
            ]

            for i, action in enumerate(actions, 1):

                st.markdown(
                    f"**{i}. {action}**"
                )

        else:

            st.success(
                "The model does not currently classify this customer "
                "as high churn risk."
            )

            st.markdown(
                """
                ### Suggested preventive actions

                - Continue monitoring customer service interactions.
                - Maintain suitable pricing and plan options.
                - Monitor changes in usage patterns.
                - Keep contract renewal information up to date.
                - Identify significant increases in service calls or charges early.
                """
            )

# ============================================================
# PAGE 2 - FEATURE IMPORTANCE
# ============================================================

elif page == "📊 Feature Importance":

    st.header("📊 Feature Importance")

    st.write(
        """
        Feature importance shows which variables the Random Forest model
        relies on most when making its predictions.

        **Important:** Feature importance indicates model dependence,
        not that a feature directly causes customer churn.
        """
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("Feature Importance Values")

    display_importance = importance.sort_values(
        "Importance",
        ascending=False
    ).copy()

    display_importance["Importance"] = (
        display_importance["Importance"] * 100
    ).round(2)

    display_importance = display_importance.rename(
        columns={"Importance": "Importance (%)"}
    )

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    top_features = (
        importance
        .sort_values("Importance", ascending=False)
        .head(3)
        ["Feature"]
        .tolist()
    )

    st.subheader("💡 Model Insights")

    st.write(
        f"""
        According to the trained Random Forest model, the three most
        important features are:

        **1. {top_features[0]}**

        **2. {top_features[1]}**

        **3. {top_features[2]}**

        These features receive relatively more importance from the model
        when distinguishing between customers who churn and those who do not.
        """
    )

# ============================================================
# PAGE 3 - MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write(
        """
        The dataset was split into training and testing sets using a
        stratified 80/20 split. The test set is used here to evaluate
        the trained model.
        """
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.3f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.3f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No Churn", "Churn"]
    )

    ax.set_yticklabels(
        ["No Churn", "Churn"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

# ============================================================
# PAGE 4 - DATASET OVERVIEW
# ============================================================

elif page == "📋 Dataset Overview":

    st.header("📋 Dataset Overview")

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        len(df)
    )

    c2.metric(
        "Churned Customers",
        int(df["Churn"].sum())
    )

    c3.metric(
        "Non-Churned Customers",
        int((df["Churn"] == 0).sum())
    )

    c4.metric(
        "Churn Rate",
        f"{df['Churn'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["No Churn", "Churn"],
        [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Distribution")

    plt.tight_layout()

    st.pyplot(fig)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )


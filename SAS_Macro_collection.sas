/****************************************************************
* PROGRAM NAME : SAS_Macro_collection_v01.sas
* DESCRIPTION : Collection of useful SAS macros, code snippets, etc.
*
* AUTHOR : ReikS
* CREATION DATE : 2023-19-22
* LAST CHANGE DATE : 2023-19-22
* REVIEWWER : <name of the reviewer>
* REVIEW DATE : <date of the review yyyy-mm-dd>
*	
*	
* SUMMARY : See below a list of the macros and code snippets contained
*           in this collection : 
*               000. Program header template
*
*
* REVIEW SUMMARY : <reviewer's notes>
* 
*
* INPUT : none
* 
* 
* OUTPUT : none
*
*
****************************************************************
* CHANGE TRACKER
* DATE			AUTHOR				DESCRIPTION
* <yyyy-mm-dd>	<name of author>	<short description>
*
****************************************************************/


****************************************************************
* 000. Program header template
****************************************************************/

/****************************************************************
* PROGRAM NAME : <fill in name of the file>
* DESCRIPTION : <fill in short description>
*
* AUTHOR : <name of the author>
* CREATION DATE : <initial creation of the file in formal yyyy-mm-dd>
* LAST CHANGE DATE : <last change of the file in yyyy-mm-dd>
* REVIEWWER : <name of the reviewer>
* REVIEW DATE : <date of the review yyyy-mm-dd>
*	
*	
* SUMMARY : <detailed summary of this program>
* 
*
* REVIEW SUMMARY : <reviewer's notes>
* 
*
* INPUT : <description of input data, files, data sources, links, etc.>
* 
* 
* OUTPUT : <description of permanent output datasets, files, tables, etc.>
*
*
****************************************************************
* CHANGE TRACKER
* DATE			AUTHOR				DESCRIPTION
* <yyyy-mm-dd>	<name of author>	<short description>
*
****************************************************************/


****************************************************************
* 900. Transfer
****************************************************************/ 


****************************************************************
* 901. Pricing
****************************************************************/ 

/**

**** Pricing in general **** ;

Pricing strategies for retail products are diverse and complex, often tailored to the specific industry, customer base, market conditions, and product characteristics. Below is an overview of some commonly used pricing strategies in the retail sector, along with sources where you can find more detailed information.

Here's an extended overview of the pricing strategies for retail products, including conditions, assumptions, advantages, disadvantages, and relevant formulas where applicable:

### 1. Cost-Plus Pricing : This is one of the most straightforward pricing methods. Retailers add a mark-up to the cost of the product to determine its selling price. This method ensures that all costs are covered and a profit margin is achieved. For more details, you can refer to "Cost-plus Pricing" on [Wikipedia](https://en.wikipedia.org/wiki/Cost-plus_pricing).

**Conditions & Assumptions**:
   - Knowledge of the product's production cost.
   - Stable cost structure.
   
**Advantages**:
   - Simple to calculate and implement.
   - Ensures that costs are covered and a profit margin is achieved.

**Disadvantages**:
   - Ignores market conditions and customer value perception.
   - May lead to overpricing or underpricing in competitive markets.

**Formula**:
   \[ Price = Cost \times (1 + Markup) \]
   Where `Cost` is the production cost, and `Markup` is the desired profit margin as a percentage.


### 2. Value-Based Pricing : This strategy involves setting prices based on the perceived value of the product to the customer, rather than on the cost of production. It requires an understanding of the customer’s needs and the value they attach to the product. The article "Value-Based Pricing" on [Harvard Business Review](https://hbr.org/2016/08/a-quick-guide-to-value-based-pricing) provides a comprehensive overview.

**Conditions & Assumptions**:
   - Understanding of customers' perceived value of the product.
   - Flexibility in pricing according to customer segments.

**Advantages**:
   - Aligns price with customer value perception.
   - Potential for higher profit margins.

**Disadvantages**:
   - Difficult to accurately gauge customer perceived value.
   - Requires continuous market research.

**Formula**: No standard formula, as it relies on qualitative assessments of value.



### 3. Competitive Pricing : Retailers set their prices based on what their competitors are charging. This strategy is common in markets with many competitors selling similar products. The "Competition-based Pricing" page on [Wikipedia](https://en.wikipedia.org/wiki/Competition-based_pricing) offers more insights.

**Conditions & Assumptions**:
   - Knowledge of competitors' pricing.
   - Similar product offerings in the market.

**Advantages**:
   - Helps to stay competitive.
   - Reduces risk of pricing too high or too low.

**Disadvantages**:
   - May lead to price wars.
   - Undermines unique value proposition.

**Formula**: No standard formula, pricing is set relative to competitors.


### 4. Dynamic Pricing : Also known as surge pricing or demand pricing, this strategy involves changing prices in real-time based on demand, competition, and other market factors. It is often used by online retailers. A detailed explanation can be found in the article "What You Need to Know About Dynamic Pricing" on [Investopedia](https://www.investopedia.com/terms/d/dynamic-pricing.asp).

**Conditions & Assumptions**:
   - Ability to monitor market demand and competitors in real-time.
   - Flexible pricing infrastructure.

**Advantages**:
   - Maximizes profits by adapting to market conditions.
   - Can quickly respond to changes in demand.

**Disadvantages**:
   - Requires sophisticated technology and data analysis.
   - May frustrate customers if prices fluctuate frequently.

**Formula**: No fixed formula, prices are adjusted based on algorithms analyzing real-time data.


### 5. Psychological Pricing : This strategy leverages psychological factors to encourage purchasing. A common example is setting prices slightly lower than a round number (e.g., $9.99 instead of $10). The "Psychological Pricing" entry on [Wikipedia](https://en.wikipedia.org/wiki/Psychological_pricing) provides more information.

**Conditions & Assumptions**:
   - Customer sensitivity to pricing.
   - Psychological triggers that influence buying behavior.

**Advantages**:
   - Can increase sales through perceived bargains.
   - Simple to implement.

**Disadvantages**:
   - Overuse may diminish effectiveness.
   - May appear gimmicky to some customers.

**Formula**: Commonly, pricing is set just below a round number (e.g., $9.99 instead of $10).


### 6. Premium Pricing : Retailers set the prices of products significantly higher than competitors to create a perception of superior quality and exclusivity. This is often seen in luxury goods. For more on this, see "Premium Pricing" on [Investopedia](https://www.investopedia.com/terms/p/premium-pricing.asp).

**Conditions & Assumptions**:
   - High-quality or unique product offerings.
   - Target market willing to pay a premium.

**Advantages**:
   - Higher profit margins.
   - Enhances brand perception as high-end or exclusive.

**Disadvantages**:
   - Limited market reach.
   - Risk of being outpriced by competitors.

**Formula**: No standard formula, prices are set significantly higher than competitors.


### 7. Promotional Pricing : This short-term strategy involves temporarily reducing prices to attract customers and increase sales volume. It is often used during sales events or for product launches. The "Sales Promotion" page on [Wikipedia](https://en.wikipedia.org/wiki/Sales_promotion) discusses this strategy in more detail.

**Conditions & Assumptions**:
   - Ability to absorb temporary reduction in margins.
   - Attractive promotional offer.

**Advantages**:
   - Boosts sales volume.
   - Attracts new customers.

**Disadvantages**:
   - May erode profit margins if overused.
   - Customers may wait for promotions to make purchases.

**Formula**: Typically involves a temporary percentage discount (e.g., 20% off).

### 8. Bundle Pricing : This involves selling multiple products together at a price lower than if they were purchased individually. This can increase the perceived value and encourage customers to buy more. The "Price Bundling" article on [Wikipedia](https://en.wikipedia.org/wiki/Price_bundling) elaborates on this approach.

**Conditions & Assumptions**:
   - Complementary products available.
   - Customer interest in purchasing multiple items.

**Advantages**:
   - Encourages customers to buy more.
   - Increases perceived value.

**Disadvantages**:
   - Reduced revenue per individual item.
   - Bundled items may not always align with customer preferences.

**Formula**:
   \[ Bundle Price = \sum Individual Prices \times (1 - Discount) \]
   Where `Individual Prices` are the prices of each product in the bundle, and `Discount` is the percentage discount applied to the bundle.

Each pricing strategy should be carefully chosen based on the specific context, market conditions, and business objectives. Combining different strategies might also be effective in addressing various market segments and achieving diverse business goals.


**** Unsecured retail loans **** ;

Unsecured retail loans' pricing typically involves accounting for the cost of funds, expected losses (due to default), operating costs, and a margin for profit. 

Sources : 
https://www.experian.com/assets/decision-analytics/white-papers/regionalization-of-price-optimization-white-paper.pdf

Given your specific context, here's an overview of pricing model types and methodologies for unsecured retail loans:

### 1. **Cost-Plus Pricing**

**Methodology**: This is the simplest method. The price (or interest rate) is set by summing up the cost of funds, the expected credit loss, the operating expenses, and the required profit margin.

**Complexity**: Low

**Challenges**: 
- Does not account for competition or market dynamics.
- May overprice or underprice loans.

**Benefits**: 
Easy to understand and implement.
Transparent for regulatory concerns.

**Further Detail**:
- **Cost of Funds**: Reflects the bank's cost to acquire the money it lends. This can include interest paid on deposits or the cost of borrowing from other institutions.
- **Expected Credit Loss (ECL)**: Estimated loss over the loan's lifetime, considering the probability of default and loss given default.
- **Operating Expenses**: Costs related to processing and managing the loan.
- **Profit Margin**: The markup added to cover the bank's profit objectives.

**Sources/References**:
- "Bank Management and Financial Services" by Peter S. Rose and Sylvia C. Hudgins provides an overview of cost-plus pricing in banking.

### 2. **Risk-Based Pricing**

**Methodology**: Interest rates are set based on the estimated probability of default (PD) of the borrower. Borrowers with higher PDs are charged higher interest rates and vice versa. 

**Complexity**: Medium

**Challenges**: 
- Requires a robust risk assessment model.
- Might be perceived as discriminatory by some customers.

**Benefits**: 
- Aligns the price with risk.
- Allows for pricing optimization.

**Further Detail**:
- **Risk Assessment**: Evaluating the borrower's probability of default (PD) using credit scores or other risk factors.
- **Pricing Adjustment**: Setting interest rates based on the assessed risk level.

**Sources/References**:
- "Credit Risk Pricing Models: Theory and Practice" by Bernd Schmid provides a comprehensive look into risk-based pricing models.
- "Pricing and Risk Management of Synthetic CDOs" by Norbert Jobst and Stavros A. Zenios (Source: Operations Research).

### 3. **Competitor-Based Pricing**

**Methodology**: Pricing is set based on competitor rates and market dynamics. The bank might set its interest rates a little below, at par, or above the competition, depending on its value proposition.

**Complexity**: Medium

**Challenges**: 
- Requires consistent monitoring of competitors.
- Reactivity might lead to a race-to-the-bottom or away from strategic objectives.

**Benefits**: 
- Stays competitive in the market.
- Can attract customers if priced correctly.

**Further Detail**:
- **Market Analysis**: Regularly reviewing competitors’ rates and adjusting prices accordingly.
- **Strategic Positioning**: Deciding whether to price loans lower, at par, or higher than competitors, based on the bank's value proposition.

**Sources/References**:
- "The Strategy and Tactics of Pricing: A Guide to Growing More Profitably" by Thomas T. Nagle and Georg Müller provides insights into competitor-based pricing strategies.


### 4. **Yield Curve Based Pricing**

**Methodology**: This method takes into account the yield curve (term structure of interest rates). Longer-term loans are often exposed to greater interest rate risk, requiring a different pricing mechanism.

**Complexity**: High

**Challenges**: 
- Need to anticipate shifts in the yield curve.
- Complex to implement.

**Benefits**: 
- Better alignment of loan pricing with market conditions.
- Accounts for maturity mismatches in bank's assets and liabilities.

**Further Detail**:
- **Interest Rate Risk**: Accounting for the risk associated with changes in interest rates over different loan maturities.
- **Yield Curve Analysis**: Using the term structure of interest rates to price loans.

**Sources/References**:
- "Interest Rate Markets: A Practical Approach to Fixed Income" by Siddhartha Jha discusses yield curve-based pricing in financial markets.


### 5. **Behavioral Pricing**

**Methodology**: Prices are set based on customer behavior insights, using advanced analytics and segmentation.

**Complexity**: High

**Challenges**: 
- Requires deep customer data analysis.
- Might be perceived as manipulative or as a privacy concern.

**Benefits**: 
- Can optimize for customer retention and acquisition.
- Enables personalization of offers.

**Further Detail**:
- **Data Analysis**: Leveraging data on customer preferences, spending habits, and interactions.
- **Segmentation**: Offering different rates based on customer segments identified through behavior analysis.

**Sources/References**:
- "Customer-Centric Pricing: The Surprising Secret for Profitability" by Utpal M. Dholakia and Itamar Simonson provides insights into behavioral pricing strategies.


### 6. **Elasticity Based Pricing**

**Methodology**: This involves adjusting prices based on demand elasticity. Prices might be adjusted based on how sensitive customers are to price changes.

**Complexity**: High

**Challenges**: 
- Requires robust demand forecasting models.
- Complex to adjust in real-time.

**Benefits**: 
- Optimizes revenue.
- Can be more responsive to market changes.

**Further Detail**:
- **Demand Forecasting**: Predicting customer reactions to price changes.
- **Dynamic Pricing**: Adjusting rates in real-time or periodically based on demand elasticity.

**Sources/References**:
- "Pricing Strategies: A Marketing Approach" by Robert M. Schindler offers an in-depth look into elasticity-based pricing.
---

Given the constraints of the bank:

- For loans with terms over 60 months and amounts exceeding 30,000 EUR, risk-based pricing combined with the yield curve-based approach would be highly effective. This ensures that the risk and term structure of interest rates are adequately accounted for.
  
- Given the DSS environment by Schufa, a simpler approach such as Cost-Plus or Risk-Based might be more feasible for actual production. However, the model development, especially the risk estimation, can be performed using sophisticated tools like SAS, Python, or R.

Finally, while simplified structures are favored, it's essential to strike a balance between simplicity and accuracy. It's crucial to align the pricing strategy with the bank's overall business objectives and risk appetite.


**** Retail loan pricing formulas **** ;

Certainly, let's proceed step-by-step.

### Step 1: Risk-Based Pricing Model

The goal of a risk-based pricing model is to set an interest rate for a loan that appropriately compensates the lender for the risks taken. The rate should cover:

1. **Risk Costs (Expected Credit Loss)**: This represents the anticipated loss from the loan due to borrower defaults. It's a function of loan amount, probability of default, loss given default, and loan term.
   
2. **Margin**: This is the profit component, ensuring the lender is adequately compensated for its services and achieves its return objectives.

3. **Funding Costs**: Representing the costs the lender incurs to secure the funds that are lent out. For a bank, this might be the interest it pays on deposits or on interbank loans.

4. **Operational Costs**: Costs associated with the origination, maintenance, and servicing of the loan throughout its term.

Given these components, the formula for the risk-based pricing rate \( R \) is:

\[ R = Risk Costs + Margin + Funding Costs + Operational Costs \]

Given our previous discussion:

1. **Risk Costs (Expected Credit Loss)**:
\[ ECL = PD \times LGD \times \left( \frac{L \times T - L \times \frac{T(T+1)}{2T}}{T} \right) \]

2. **Margin**:
To be derived later using RARORAC.

3. **Funding Costs**: 
Let's denote this as \( C \) (constant per annum for the entire loan term due to hedging).

4. **Operational Costs**: 
Can be denoted as \( OC \). This can be a fixed value or a percentage of the loan amount.

Combining these, the rate \( R \) is:

\[ R = ECL + Margin + C + OC \]


Now, let's implement this in Python:

Note: The `Margin` parameter in this function is a placeholder, and its calculation using RARORAC will be detailed in the next steps.

```python

def risk_based_pricing_rate(PD: float, LGD: float, L: float, T: int, C: float, OC: float, Margin: float) -> float:
    """
    Calculate the risk-based pricing rate for a loan.
    
    Arguments:
    PD (float)      : Probability of Default for the borrower.
    LGD (float)     : Loss Given Default, represents the portion of the loan that will be lost if a default occurs.
    L (float)       : Loan amount.
    T (int)         : Loan term in months.
    C (float)       : Funding cost, the cost the bank incurs to secure the funds lent out (per annum).
    OC (float)      : Operational cost associated with the origination, maintenance, and servicing of the loan (per annum).
    Margin (float)  : Profit component to ensure the lender is compensated for its services.
    
    Returns:
    R (float)       : The interest rate set for the loan to cover risk, funding and operational costs, as well as the desired margin.
    """
    
    ECL = PD * LGD * (L * T - L * (T*(T+1))/(2*T)) / T
    R = ECL + Margin + C + OC
    return R

# Testing Facility:
if __name__ == "__main__":
    # Sample test values
    PD = 0.03   # 3% probability of default
    LGD = 0.5  # 50% loss given default
    L = 10000  # Loan amount of 10,000 EUR
    T = 60     # Loan term of 60 months (5 years)
    C = 0.02   # 2% funding cost per annum
    OC = 0.01  # 1% operational cost per annum
    Margin = 0.03  # 3% margin for profit
    
    # Calculate and print the risk-based pricing rate
    R = risk_based_pricing_rate(PD, LGD, L, T, C, OC, Margin)
    print(f"The risk-based pricing rate for the loan is: {R:.2%}")

```

The Capital Adequacy Ratio (CAR) is a measure used by banks to determine the adequacy of their capital keeping in view their risk exposures. It represents the proportion of a bank's capital to its risk-weighted assets. This ratio ensures that banks have enough capital on reserve to absorb a reasonable amount of loss before becoming insolvent.
Regulatory bodies, such as the Basel Committee on Banking Supervision, set minimum CAR standards to ensure that banks can absorb a reasonable amount of loss. For instance, under the Basel III standards, the minimum CAR is set at 8%, though individual countries' banking regulators may require higher levels.
However, while there is a regulatory minimum, individual banks often set a higher internal CAR based on their risk appetite, strategic objectives, and market conditions. This internal CAR is indeed a strategic choice by the bank. Banks often maintain a CAR above the regulatory minimum to ensure a buffer against unforeseen losses and to convey financial robustness to investors and customers.

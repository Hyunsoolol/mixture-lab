<style>
/* 1. 슬라이드 기본을 다 왼쪽 정렬로 강제 */
.reveal .slides > section,
.reveal .slides > section section {
  text-align: left !important;
}

/* 2. 제목들도 왼쪽 정렬 */
.reveal .slides section h1,
.reveal .slides section h2,
.reveal .slides section h3 {
  width: 100% !important;
  display: block !important;
  text-align: left !important;
  margin-left: 0 !important;
  margin-right: auto !important;
}
.reveal .slides section h4 {
  width: 100% !important;
  display: block !important;
  text-align: left !important;
  margin-left: 0 !important;
  margin-right: auto !important;
}

/* 3. 문단/목록도 왼쪽 정렬 */
.reveal p,
.reveal ul,
.reveal ol,
.reveal li {
  text-align: left !important;
}

/* 4. KaTeX 블록 수식도 왼쪽으로 붙이기 */
.reveal .katex-display {
  font-size: 0.9em;
  text-align: left !important;
  margin-left: 0 !important;
}

/* 5. 인라인 수식 크기 조금 줄이기 */
.reveal .katex-inline {
  font-size: 0.8em;
}

/* 6. 전체 글자 크기(조금 작게) */
.reveal .slides section {
  font-size: 0.6em;
}

/* 7. 제목 폰트 크기 조정 */
.reveal h1 {
  font-size: 1.4em;
  margin-bottom: 0.4em;
}

.reveal h2 {
  font-size: 1.2em;
  margin-bottom: 0.3em;
}

.reveal h3 {
  font-size: 1.1em;
  margin-bottom: 0.2em;
}

.reveal h4 {
  font-size: 0.8em;
  margin-bottom: 0.2em;
}

/* --- Mermaid 다이어그램 전용 강제 확대 설정 --- */
.reveal .mermaid {
  text-align: center !important;  /* 그림 중앙 정렬 */
  width: 100% !important;         /* 너비 꽉 채우기 */
  margin: 10px auto !important;   /* 위아래 여백 */
}

.reveal .mermaid svg {
  width: 80% !important;          /* SVG 크기 80%로 설정 (너무 크면 100%로 변경) */
  height: auto !important;        /* 높이 자동 */
  max-width: none !important;     /* 너비 제한 해제 */
  min-height: 500px !important;   /* 최소 높이 확보 */
}

/* 박스 안의 텍스트 정렬 */
.reveal .mermaid .node foreignObject,
.reveal .mermaid .node div {
  text-align: center !important; 
  line-height: 1.3 !important;
  font-size: 8px !important;     /* 글자 크기 강제 고정 */
}

</style>
---
## 1. Title & Authors

### Mixture Conditional Regression with Ultrahigh Dimensional Text Data for Estimating Extralegal Factor Effects

**Authors:** Jiaxin Shi, Fang Wang, Yuan Gao, Xiaojun Song, and Hansheng Wang 

**Journal:** _The Annals of Applied Statistics (2024)_

---

## 2. Motivation & Problem Definition  

### Research Goal
- **Objective:** 사법적 의사결정(Judicial Decision)에서 인종, 성별 등 **Extralegal Factors**($X_1$)가 판결($Y$)에 미치는 순수한 인과 효과(Causal Effect) 추정.
    
- **Challenge:** 판결문 텍스트와 같은 **Legal Factors**($Z$)를 통제해야 하는데, 이는 **Ultrahigh-dimensional Data** ($p \gg n$)임.

### Limitations of Existing Methods
- **Standard Regression (OLR):** $p > n$ 상황에서 작동 불가.
    
- **Regularization (Lasso, SCAD):** 변수 선택 과정에서 **Omitted Variable Bias** 발생 가능성.
    
- **Naive Approach:** 텍스트 데이터를 무시하거나, 임의로 차원 축소 시 정보 손실 발생.

---

## 3. Methodology: MCR Model Structure
### Model Specification

전체 데이터를 $K$개의 **Latent Classes**로 가정하는 **Mixture Model** 프레임워크 제안.

### 1) Response Model (Conditional on Class $k$)

$$Y_i = \gamma_k + X_i^T \theta + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)$$

- **Heterogeneity:** 절편($\gamma_k$)은 클래스별로 다르지만, 관심 변수의 효과($\theta$)는 **Global Parameter**로 고정.
    

### 2) Feature Model (Naive Bayes Assumption)

$$P(Z_i | K_i = k) = \prod_{j=1}^{p} p_{kj}^{Z_{ij}} (1 - p_{kj})^{1 - Z_{ij}}$$

- **Dimensionality:** $Z_i$는 $p$-차원 이진 벡터 (Binary Text Features).
    
- **Assumption:** 클래스가 주어졌을 때, 단어들은 서로 **Conditionally Independent**하다고 가정.

---

## 4. Theoretical Insight: Blessing of Dimensionality
### Curse vs. Blessing

- 일반적으로 고차원은 통계적 추론의 적(**Curse of Dimensionality**)으로 간주됨.
    
- 하지만 본 MCR 모형에서는 $p \to \infty$일수록 **Latent Class Identification**이 완벽해짐.

### Key Theorems

- Theorem 2 (Identification Consistency):
    
    $$P(\max_{i} |\hat{\pi}_{ik} - I(K_i=k)| > e^{-\nu p}) \to 0$$
    
    - $p$가 커질수록 사후 확률(Posterior Probability) 오차가 **Exponential Rate**로 감소.
        
- **Result:** 고차원 텍스트 정보가 많을수록 어떤 범죄 유형(Class)인지 명확히 구분 가능 $\rightarrow$ 정확한 통제(Control) 가능.

---

## 5. 4-step Estimation Pipeline (The Core Algorithm)
### Pseudo-EM Algorithm Flow

기존의 단순 EM 알고리즘은 파라미터 과다로 수렴이 어렵습니다. 이를 해결하기 위한 독창적인 4단계 추정법입니다.

![[Pasted image 20260112121824.png]]

1) **Initial Estimator ($\hat{\Omega}$):** $Z$를 제외하고 $Y, X$만으로 초기 파라미터 추정. ($\sqrt{n}$-consistency 확보)
2) **Response Prob Estimator ($\hat{p}_{kj}$):** 각 클래스별 단어 발생 확률 추정.
3) **Class Membership ($\hat{\pi}_{ik}$):** $Z$의 고차원 정보를 결합하여 사후 확률 계산. (**High Accuracy**)
4) **Final Estimator ($\hat{\Omega}_{real}$):** $\hat{\pi}_{ik}$를 가중치로 사용하여 최종 $\theta$ 추정. (**Oracle Property 달성**)

---

## 6. Empirical Analysis (Real Data)

### 🇨🇳 Data Description & Latent Classes

- **Dataset:** Chinese Burglary Cases (2017-2018) from CJO1.
    
    - Sample size $n = 6,118$.
        
    - Response $Y$: Log-transformed sentence length.
        
    - Text Features $Z$: $p = 6,514$ keywords from judgment documents.
        
- **Latent Class Discovery ($K=7$ selected by BIC):**
    
    - MCR 모델은 텍스트($Z$)를 이용해 범죄의 **세부 유형(Nature of Crime)**을 성공적으로 군집화함2.
        
    - **Class 1:** 주거 침입 (Home, break-in).
        
    - **Class 2:** 고가 전자기기 절도 (iPhone, Laptop).
        
    - **Class 4:** 귀금속 절도 (Gold, Necklace, Ring).
        
	    $\rightarrow$ _Text semantics reveal qualitative differences in crimes._

---
### Estimation Results: MCR vs. OLR

**Key Question:** 성별(Gender)이 형량에 영향을 미치는가? (Testing Judicial Impartiality) 

|**Variable (X)**|**OLR (Traditional)**|**MCR (Proposed)**|**Interpretation**|
|---|---|---|---|
|**Gender (Male)**|**0.1081** ($p=0.025$)|**0.0662** ($p=0.088$)|**Bias Correction**|
|**Ethnicity (Han)**|-0.0761 ($p=0.000$)|-0.0633 ($p=0.000$)|Robust|
|**Age**|0.0016 ($p=0.032$)|0.0012 ($p=0.042$)|Robust|

- **OLR Result:** 남성이 여성보다 유의하게 형량이 높음 $\rightarrow$ **Potential Gender Bias?**
    
- **MCR Result:** 범죄 유형(Latent Class)을 통제하자 **성별 효과는 유의하지 않게 됨 ($p > 0.05$).**
    
- **Conclusion:** 남성이 더 중형이 선고되는 범죄 유형(예: 강도 등)에 연루될 확률이 높았던 것임. MCR은 이러한 **Omitted Variable Bias**를 효과적으로 제거함.

---
### Model Performance (Prediction)

- **Out-of-sample Prediction ($R^2$):**
    
    - 데이터를 Training(50%) / Testing(50%)으로 분할하여 검증.
        
    - **MCR:** 48.10% (SD 1.24%)
        
    - **OLR:** 42.91% (SD 1.20%)
        
- **Result:**
    
    - MCR이 OLR보다 약 **5.2% 포인트 더 높은 설명력**을 보임.
        
    - Two-sample t-test 결과 차이가 매우 유의함 ($p < 10^{-8}$).
        
    - 초고차원 텍스트 데이터($Z$)를 활용하는 것이 모형의 **Predictive Power**와 **Interpretability**를 모두 향상시킴.
---
### Interpretation of Latent Classes

TF-IDF 상위 키워드 기반 의미 부여

| Class (표본수) | Top keywords(예시)                                     | 해석 요약      |
| ----------- | ---------------------------------------------------- | ---------- |
| 1 (2809)    | home, breakin, record, window, climb                 | 전과/침입 정황   |
| 2 (1265)    | photo, cigarette, laptop, video, apple               | 담배·전자기기 절도 |
| 3 (807)     | largeramount, possession, secrecy, indemnity, weapon | 고액/중대 정황   |
| 4 (763)     | gold, necklace, ring, bracelet, watch                | 귀금속 절도     |
| 5 (288)     | cash, phone, bedroom, gate, wardrobe                 | 현금·생활공간 절도 |
| 6 (168)     | liability, drive, force, motorcycle, joint           | 공범/결합 범행   |
| 7 (18)      | knife, robbery, violence, escape, threat             | 강도/폭력 전환   |

- 범죄의 심각성(Severity) 수법(Modus Operandi)을 나타내는 핵심 지표임을 입증함.
- 특히 Class 7(강도)이나 Class 6(공동 범죄) 같이 형량에 큰 영향을 미치는 요인들을 별도 그룹으로 정확히 분리해냄.
---

## 7. Conclusion & Discussion

### Summary

- **Methodology:** 고차원 텍스트 데이터를 활용해 교란 요인을 완벽히 통제하는 **Mixture Conditional Regression** 개발.
    
- **Theory:** 고차원이 잠재 집단 식별을 돕는 **"Blessing of Dimensionality"** 증명.
    
- **Efficiency:** Oracle Estimator와 점근적으로 동일한 효율성 입증.
    

### Future Works (Related to My Research)

- **Relaxing Assumptions:** Binary $Z$ 대신 **Frequency**나 **Embedding** 활용 가능성.
    
- **Model Extension:** Linear Regression을 넘어 GLM (Logistic, Poisson)으로의 확장.
    
- **Optimization:** EM 알고리즘의 Global Convergence에 대한 이론적 보완.
---


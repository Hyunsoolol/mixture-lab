## 1. 개요 (Overview)

본 연구는 Kim et al. (2005)이 제안한 비모수적 검정(Log-rank test)의 핵심 아이디어를 준모수적 회귀모형인 **Cox Proportional Hazards Model**로 확장하는 것을 목표로 한다.

특히 기존의 구간 중도절단(Interval Censored, IC) 데이터 분석에서 필수적으로 여겨지던 EM 알고리즘(Iterative Expectation-Maximization)의 반복 연산을 제거하여, 계산 복잡도(Computational Complexity)를 획기적으로 낮추는 데 초점을 맞춘다. 이는 추후 SIMEX(Simulation Extrapolation)와 같이 반복적인 모델 적합이 요구되는 분석 환경에서 필수적인 계산 효율성을 보장한다.

## 2. 기본 설정 및 기호 정의 (Notation and Setup)

$n$명의 독립적인 개체에 대하여, 실제 생존 시간(True Failure Time)을 $T_i$ ($i=1, \dots, n$)라고 하자. Cox 비례위험모형 가정에 따라 $T_i$의 위험 함수(Hazard Function)는 공변량 벡터 $\mathbf{Z}_i$에 의존한다.

$$h(t | \mathbf{Z}_i) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{Z}_i)$$

여기서 $h_0(t)$는 기저 위험 함수(Baseline Hazard Function)이며, $\boldsymbol{\beta}$는 우리가 추정하고자 하는 회귀계수 벡터이다.

### 2.1 구간 중도절단 (Interval Censoring)

데이터는 정확한 $T_i$ 대신 관측 구간 $(L_i, R_i]$ 형태로 주어진다.

$$T_i \in (L_i, R_i]$$

- $L_i$: 관측 구간의 하한 (Lower Bound)
    
- $R_i$: 관측 구간의 상한 (Upper Bound)
    
- **Right Censored:** 만약 $R_i = \infty$라면, 이는 우측 중도절단(Right Censored) 데이터로 취급한다.


## 3. 방법론적 확장 (Methodological Derivation)

### 3.1 시간축의 이산화 (Discretization of Time)

Cox 모형의 부분 우도(Partial Likelihood)를 구성하기 위해 연속형 시간축을 사건이 발생 가능한 이산적인 시점들로 변환한다.

모든 관측치 구간의 유한한 끝점들 $\bigcup_{i=1}^n {L_i, R_i} \cap [0, \infty)$을 수집하고 정렬하여, 유니크한 사건 발생 후보 시점 집합 $\mathcal{S}$를 정의한다.

$$\mathcal{S} = \{ s_1, s_2, \dots, s_m \} \quad (\text{단, } s_1 < s_2 < \dots < s_m)$$

### 3.2 가중치 행렬 $\mathbf{W}$의 구성 (The Core Innovation: EM-free Weights)

기존 방법(Turnbull 등)은 각 구간 내에서 확률 질량이 어떻게 분포하는지 모르기 때문에, 이를 모수 $\boldsymbol{\theta}$로 두고 EM 알고리즘을 통해 반복 추정한다.

$$w_{ij}^{(k)} \propto S^{(k)}(s_j) - S^{(k)}(s_{j+1}) \quad (\text{매 단계마다 갱신})$$

하지만 **Kim et al. (2005)**은 귀무가설 하에서 혹은 계산 효율성을 위해 **"구간 내 사건 발생 확률은 균등하다(Uniform)"**는 가정을 도입한다. 이에 따라 $i$번째 개체가 특정 시점 $s_j$에서 사건을 경험했을 확률 가중치 $w_{ij}$는 **상수(Constant)**로 고정된다.

$$w_{ij} = \Pr(T_i = s_j | T_i \in (L_i, R_i]) \approx \begin{cases} \frac{1}{m_i} & \text{if } s_j \in (L_i, R_i] \\ 0 & \text{otherwise} \end{cases}$$

여기서 $m_i$는 $i$번째 개체의 구간 내에 포함된 후보 시점의 총 개수이다.

$$m_i = \sum_{k=1}^m I(s_k \in (L_i, R_i])$$

이 $w_{ij}$는 데이터가 주어지면 즉시 계산 가능하며, $\boldsymbol{\beta}$ 추정 과정에서 변하지 않는다. 이것이 **EM-free 접근법의 핵심**이다.

### 3.3 리스크 셋의 정의 (Definition of Risk Set)

시점 $s_j$에서의 리스크 셋(Risk Set) $\mathcal{R}(s_j)$는 **"시점 $s_j$ 직전까지 생존해 있었을 가능성이 있는 모든 개체의 집합"**으로 정의된다.

구간 중도절단 데이터에서 개체 $i$가 $s_j$ 시점에 리스크 셋에 포함되려면, 해당 개체의 구간 상한 $R_i$가 $s_j$보다 크거나 같아야 한다. (즉, $s_j$ 이전에 사건이 발생했음이 확정되지 않아야 한다.)

$$\mathcal{R}(s_j) = \{ k : R_k \ge s_j \}$$

### 3.4 수정된 가중 부분 우도 함수 (Modified Weighted Partial Likelihood)

이제 Cox 모형의 부분 우도 함수를 $w_{ij}$를 반영하여 재구성한다.

일반적인 Cox 모형의 우도 함수는 $L(\boldsymbol{\beta}) = \prod \frac{\theta_i}{\sum \theta_k}$ 형태이다. 여기서 우리는 시점 $s_j$마다 **"한 명의 확실한 사건"**이 발생하는 것이 아니라, **"여러 명의 개체가 각각 $w_{ij}$만큼의 확률적 기여"**를 한다고 가정한다. 이를 위해 Breslow의 동점(Tie) 처리 방식을 확장하여 적용한다.

시점 $s_j$에서의 우도 기여분(Likelihood Contribution) $L_j(\boldsymbol{\beta})$는 다음과 같다.

$$L_j(\boldsymbol{\beta}) = \frac{\prod_{i=1}^n \left( \exp(\mathbf{Z}_i^\top \boldsymbol{\beta}) \right)^{w_{ij}}}{\left( \sum_{k \in \mathcal{R}(s_j)} \exp(\mathbf{Z}_k^\top \boldsymbol{\beta}) \right)^{D_j}}$$

여기서 $D_j$는 시점 $s_j$에서 발생한 것으로 기대되는 **가중된 사건의 총량(Expected Number of Events at $s_j$)**이다.

$$D_j = \sum_{i=1}^n w_{ij}$$

위 $L_j(\boldsymbol{\beta})$를 모든 시점 $j=1, \dots, m$에 대해 곱하여 전체 부분 우도 함수(Overall Partial Likelihood)를 얻는다.

$$\mathcal{L}_{mod}(\boldsymbol{\beta}) = \prod_{j=1}^m \frac{\exp \left( \sum_{i=1}^n w_{ij} \mathbf{Z}_i^\top \boldsymbol{\beta} \right)}{\left( \sum_{k \in \mathcal{R}(s_j)} \exp(\mathbf{Z}_k^\top \boldsymbol{\beta}) \right)^{D_j}}$$

### 3.5 로그-부분 우도 함수 (Log-Partial Likelihood Function)

최적화를 위해 위 식에 자연로그를 취하면, 우리가 최대화해야 할 목적 함수(Objective Function) $\ell_{mod}(\boldsymbol{\beta})$가 도출된다.

$$\begin{aligned} \ell_{mod}(\boldsymbol{\beta}) &= \log \mathcal{L}_{mod}(\boldsymbol{\beta}) \\ &= \sum_{j=1}^m \left[ \underbrace{\sum_{i=1}^n w_{ij} \mathbf{Z}_i^\top \boldsymbol{\beta}}_{\text{Term A}} - \underbrace{\left( \sum_{i=1}^n w_{ij} \right) \log \left( \sum_{k \in \mathcal{R}(s_j)} \exp(\mathbf{Z}_k^\top \boldsymbol{\beta}) \right)}_{\text{Term B}} \right] \end{aligned}$$

- **Term A (가중된 공변량의 합):** 시점 $s_j$를 포함하는 구간을 가진 개체들이, 그 확률($w_{ij}$)만큼 공변량 효과에 기여하는 항이다.
    
- **Term B (리스크 셋 보정 항):** 시점 $s_j$에서의 총 위험도(Total Hazard in Risk Set)를 해당 시점의 총 사건 발생 가중치($D_j$)만큼 반영하여 차감하는 항이다.
    

## 4. 추정 및 통계적 성질 (Estimation and Inference)

### 4.1 스코어 함수 (Score Function)

회귀계수 $\hat{\boldsymbol{\beta}}$을 찾기 위해 $\ell_{mod}(\boldsymbol{\beta})$를 $\boldsymbol{\beta}$에 대해 1차 미분한 스코어 함수 $U(\boldsymbol{\beta})$를 구한다.

$$U(\boldsymbol{\beta}) = \frac{\partial \ell_{mod}}{\partial \boldsymbol{\beta}} = \sum_{j=1}^m \left[ \sum_{i=1}^n w_{ij} \mathbf{Z}_i - D_j \frac{\sum_{k \in \mathcal{R}(s_j)} \mathbf{Z}_k \exp(\mathbf{Z}_k^\top \boldsymbol{\beta})}{\sum_{k \in \mathcal{R}(s_j)} \exp(\mathbf{Z}_k^\top \boldsymbol{\beta})} \right]$$

여기서 분수 항은 시점 $s_j$에서의 **가중 평균 공변량(Weighted Average Covariate in Risk Set)** $\bar{\mathbf{z}}(s_j, \boldsymbol{\beta})$을 의미한다. $U(\boldsymbol{\beta}) = \mathbf{0}$을 만족하는 해는 기존 Kim et al. (2005)의 로그 순위 검정 통계량이 0이 되는 조건과 일치한다.

### 4.2 정보 행렬 (Information Matrix)

분산 추정을 위해 2차 미분한 헤시안 행렬(Hessian Matrix) $I(\boldsymbol{\beta})$를 구한다.

$$I(\boldsymbol{\beta}) = -\frac{\partial^2 \ell_{mod}}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^\top}$$

이때, 추정된 파라미터 $\hat{\boldsymbol{\beta}}$의 공분산 행렬은 정보 행렬의 역행렬인 **Naive Variance Estimator**로 근사할 수 있다.

$$\text{Var}(\hat{\boldsymbol{\beta}}) \approx I^{-1}(\hat{\boldsymbol{\beta}})$$

> **Note:** 본 모형은 한 개체의 사건 확률을 여러 시점에 분산($w_{ij}$)시켜 처리하므로, 데이터의 상관성을 고려하지 않은 Naive Variance는 실제 분산을 과소 추정할 가능성이 있다. 따라서 보다 엄밀한 통계적 추론이나 SIMEX 적용 시에는 **부트스트랩(Bootstrap)** 또는 **샌드위치 분산 추정량(Robust Sandwich Variance Estimator)**의 사용이 권장된다.


## 5. 기존 방법론과의 비교 및 의의 (Conclusion)

|**구분**|**기존 방법 (ICM / EM-based)**|**제안된 방법 (Uniform Weight IC Cox)**|
|---|---|---|
|**핵심 접근**|생존함수 $S(t)$를 반복적으로 갱신하며 우도 최대화|Risk Set 크기에 기반한 **고정 가중치(Fixed Weight)** 사용|
|**가중치 성격**|$\boldsymbol{\beta}$와 $S(t)$에 의존하는 변수 (Variable)|구간 길이 및 밀도에 의존하는 상수 (Constant)|
|**최적화 과정**|**Nested Loop:** Optim 내부에 EM Loop 존재|**Single Loop:** 일반적인 Newton-Raphson 1회 수행|
|**계산 속도**|매우 느림 (수렴 속도에 의존)|**매우 빠름 (Standard Cox와 유사)**|

본 방법론은 **Kim et al. (2005)의 "Risk Set 기반 가중치" 아이디어**를 **Cox 모형의 Score Function 구조**에 완벽하게 이식함으로써, 계산 복잡도를 획기적으로 낮추었다. 이는 특히 계산 비용이 높은 **SIMEX (Simulation Extrapolation)** 알고리즘을 적용하여 측정 오차(Measurement Error)를 보정해야 하는 본 연구의 2차 목표를 달성하기 위한 가장 효율적이고 강력한 베이스라인 모델이 된다

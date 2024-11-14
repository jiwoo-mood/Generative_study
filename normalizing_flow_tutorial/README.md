# Normalizing_flow


$$p(x)=p_{z}(f(x))*|\det Df(x)|$$
 <br> 
  <br> 
- 이전까지 정확한 확률 밀도 (likelihood) 계산하기 어려웠다.
- Det 텀을 통해 매핑되는 확률을 normalizing 하여 합 1이 되도록 한다.
- $p_x(x)$ 를  $p_z(z)$ 로 변환하는 $f(x)$ 를 학습시키겠다.
- $p_z(z)$은 N(0,I) 인 노이즈 분포를 따른다.
<br> 
- f(x)조건
    1. 역함수가 존재해야 한다. (invertible_ 1:1 대응)
    2. 미분 가능해야 한다.
    3. Jacobian Determinant과 역행렬에 대해 효율적으로 계산 가능해야 한다. (추적 가능성)
 <br> 
 <br> 
 
  **Forward** (data → latent space)

    1. 데이터를 간단한 분포(정규 분포)에서 모델링하기 쉽게 변환한다. 
    2. x를 f1, f2, …fk transformation을 통해 latent space로 매핑한다.
    3. 각 transformation 마다 density estimation을 수행하여 확률 밀도 변화를 추적한다. 이를 통해 원래 데이터 분포의 px(x) density를 계산할 수 있다.
    4. z는 pz(z)를 따르게 됨.
 
   <br>  <br> 
    **Backward** (latent space → data)
    
    1. pz(z)에서 z를 샘플링한 후, 역함수 f를 적용해서 z를 x로 매핑
    2. inverse transformation은 fk, fk-1 , … f1 순인 역순으로 적용된다.
    3. 최종적으로 샘플링된 z가 복잡한 데이터 분포로 변환되어 x로 나타나고, 새로운 샘플이 된다.
<br>
<br>

### 1-2. Flows 종류

1. Linear Flows
    
    $f(x) = Ax +b$
    
    Inverse : $f^{-1}(z) = A^{-1}(z-b)$
    
    Determinant : $\det Df(x) = \det A$
    
    표현력이 떨어지며 $O(d^3)$ 라는 계산 비용을 가짐.
<br>

2.Coupling Flows
    

    
    - 여기서 Affine function을 추가하면 Affine Coupling flows가 된다. Affine Coupling flows는 선형적인 곱셈/덧셈만 사용하여 Jacobian을 쉽게 계산하도록 한다.
    - Recursive Coupling Flows
        

<br>     
3.Autoregressive Models as flows
    
  Autoregressive 를 flow 방식으로 활용할 수 있다.
  
  $p(x) = \prod_{t=1}^{D}{p(x_i|x_<i)}$
  
  $p(xi|x_<i) = N(x_i | \mu (x_<i), \sigma (x_<i))$
  
  $f_i(x) = \frac{x_i - \mu(x_{<i})}{\sigma (x_{<i})}$
  
  $\det Df(x) = \prod_i \sigma ^{-1}(x_{<i})$
  
  Inverse : $f^{-1} _i (z) = \frac{z_i - \mu(z_{<i})}{\sigma(z_{<i})}$
    
    - 샘플링이 순차적이고 느리지만 Density evaluation은 병렬적으로 수행된다.
<br>
4. Multi Scale flows
    
  the flow preserves dimensionality, but this is expensive in high dimensions.
  
  실질적으로는 dimensions를 dropping하면서 사용
  
  - Coupling flow의 한 종류가 될 수도 있긴 하나, dropped dimensions를 추적해야 하는 것이 중요하다. (invertible)
  - image를 다룰 땐 Squeeze를 사용해서 차원을 줄이고 채널을 얻는 식으로 dimension을 쪼갠다.
  - 이렇게 함으로써 다양한 scale의 정보를 계층적으로 표현될 수 있음. 초기 단계에서는 low 정보를 후반부에서는 high 정보를 학습하는 등의 복잡한 분포를 효과적으로 학습할 수 있음.
  - 계산 효율을 높이고 메모리 효율이 좋아짐.
<br>
<br>
### 1-3. Discrete time NF → Continuous time NF

- discrete을 continuous으로 모델링을 하고 싶다 → ODE를 사용해라, Dequantization을 사용해라!

1. ODE를 사용해라! (예시로, FFJORD)
    
    - discrete : $px(x) = pz(f(x))|\det Df(x)|$
    
    - ODE : $f(x) = y_o + \int _0 ^1 {h(t, y_t)}dt \text { with y0 = x}$
    
    - h = $\int_0 ^1 {Tr(\frac{\partial h}{\partial y}(t,y_t))}dt$ (Hutchinson Trace Estimator)
    
    - 따라서 완성된 Continuous 
    
    - $$\log p_x(x) = \log p_z(f(x)) + \int_0 ^1 {Tr(\frac{\partial h}{\partial y}(t,y_t))}dt$$
    

2. Dequzntization을 사용해라!
    - pixel intensities는 보통 discrete or quantized 하다.
    - 보통 discrete data가 많고, 이를 continuous model에 넣는 것은 singularities (불연속/불해석적) 이다.
    - 따라 Noise를 더해서 Continuous data로 접근하겠다.
    - discrete 분포에 continuous한 노이즈 u를 더해서 추가해서 continuous하게 만든다.
    - q(v)가 p(v)에 근사하도록 학습한다.

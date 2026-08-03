# 건양대학교 lab-corps 과제🌳: Factory Automation System🏭

반도체 공장 자동화 시스템에서 롤러의 간격을 자동화 하는 시스템.

## 🧑‍💻Code(비선형 회귀 모델)
**AutomationSystemCode.py**
> spicy.optimize 에서 minimize를 사용해서 회귀함수를 구하는 코드. costfuntion은 mse를 사용.
> 최적 파라미터를 저장함

**TalkFile**
> 데이터 csv 파일

**data.py**
> 데이터 불러오기 및 minmaxscaler로 전처리

**TestCode.py**
> 저장된 파라미터를 불러와서 테스트함.
---

초기 데이터는 밑이 0<x<1인 지수함수 모양으로 데이터들이 분포해 있음. 그래서 회귀 함수의 모양도 밑이 분수인 지수함수의 모양으로 나올것으로 예상.

## Result

### Train 
ws_gap_diff result

> p0(초기값): [0.001, 0.001, 0.001]
> a=0.5557, b=-6.2279, c=0.1378
> MSE: 0.0111


ds_gap_diff result

> p0(초기값): [0.001, 0.001, 0.001]
> a=0.6260, b=-6.3376, c=0.1579
> MSE: 0.0102

### Test
ws_gap_diff model

> ws params: [ 0.55572231 -6.22794291  0.13776184]
> MSE: 0.0074

ds_gap_diff model

> ds params: [ 0.62596188 -6.33760715  0.15785487]
> MSE: 0.0119

## 총정리
![original data](https://github.com/user-attachments/assets/f4fd04e8-91d0-4d25-9833-48a4f8e2229d)
> 초기 데이터 분포. 전체적으로 감소지수함수 형태의 모양을 하고있지만, y값들의 분포 차이가 커서 노이즈가 심함.

![scaled data](https://github.com/user-attachments/assets/3757298b-2792-4b52-b7d7-6c7814eca2aa)
> MinMaxScaler로 데이터 전체를 스케일링한 분포. 범위 자체만 0~1 사이의 값으로 바꿔서 데이터의 의미는 유지됨.

![fitting model with test data](https://github.com/user-attachments/assets/48bc30d9-2f24-4fde-b1d3-ba42a5b5f1fb)
> 피팅된 모델 그래프와 테스트 데이터 분포를 나타낸 그래프. 전체적으로 추세는 잘 잡았으나 데이터 자체에 y값 노이즈가 심해서 간단한 비선형 함수로는 좋은 예측이 불가함.

---

결론적으로 데이터 자체의 **노이즈**가 심해서 비선형회귀 모델로는 피팅이 불가함.

MSE값이 낮게 나온건 데이터를 스케일링 해서 낮게 나온것.

테스트 기준 ws RMSE: √0.0074 ≈ 0.086 → 역변환 ±86, ds RMSE: √0.0119 ≈ 0.109 → 역변환 ±109

평균적으로 86~109 정도 오차가 있음.

## 머신러닝 모델
XGB, lightgbm, randomforest, MLP 총 4가지 모델 사용.

전처리는 X 데이터에만 StandardScaler로 사용.

![machine learning result](https://github.com/user-attachments/assets/7834f58d-b1db-4bb2-a429-2a97daaeb24c)
> xgboost가 성능적으로 제일 잘 나옴.

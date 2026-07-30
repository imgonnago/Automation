# 건양대학교 lab-corps 과제🌳: Factory Automation System🏭

반도체 공장 자동화 시스템에서 롤러의 간격을 자동화 하는 시스템.

## 🧑‍💻Code
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
> a=2.0001, b=-0.6496, c=-1.2391
> MSE: 0.0365


ds_gap_diff result

> p0(초기값): [0.001, 0.001, 0.001]
> a=0.0913, b=2.4640, c=0.0369
> MSE: 0.0138

### Test
ws_gap_diff model

>ws params: [ 2.00014131 -0.64956346 -1.239068  ]
>MSE: 0.0263

ds_gap_diff model

>ds params: [0.09134144 2.46399558 0.03687869]
>MSE: 0.0110

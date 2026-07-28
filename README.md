# Factory Automation System

반도체 공장 자동화 시스템에서 롤러의 간격을 자동화 하는 시스템.

## Code
**AutomationSystemCode.py**
> spicy.optimize 에서 minimize를 사용해서 회귀함수를 구하는 코드. costfuntion은 mse를 사용

**TalkFile**
> 데이터 csv 파일

**data.py**
> 데이터 불러오기 및 minmaxscaler로 전처리

---

초기 데이터는 밑이 0<x<1인 지수함수 모양으로 데이터들이 분포해 있음. 그래서 회귀 함수의 모양도 밑이 분수인 지수함수의 모양으로 나올것으로 예상.

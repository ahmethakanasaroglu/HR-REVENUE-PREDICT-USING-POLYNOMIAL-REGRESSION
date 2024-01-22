#%% Polynomial Linear Regression
#   y = a + b1x + b2x^2 + b3x^3 + b4x^4 + ..... + bN*x^N

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("C:\\Users\\asaro\\Downloads\\polynomial.csv",sep=';')


# plot dataset
plt.scatter(df['deneyim'],df['maas'])
plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')
plt.savefig('1.png', dpi=300)
plt.show()

# veriler görüldüğü gibi dogrusal bir yapıda dagılmıyor
# eger biz bu veri setine linear regression uygularsak hiç uygun olmayan bir tahmin çizgisi görürüz

reg = LinearRegression()
reg.fit(df[['deneyim']],df['maas'])

plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')

plt.scatter(df['deneyim'],df['maas'])

xekseni = df['deneyim']
yekseni = reg.predict(df[['deneyim']])
plt.plot(xekseni,yekseni,color="green",label="linear regression")
plt.legend()
plt.show()

# Bir adet polynomial regression nesnesi oluşturması için PolynomialFeatures fonksiyonu cagırıyoruz.
# Bu fonksiyonu çağırırken polinomun derecesini (N) belirtiyoruz.
# x degerimizi polinom yukardaki fonksiyonuna uyacak şekilde uyarlanmasını sağlıyoruz.  1,x,x^2(n=2) şeklinde
polynomial_regression = PolynomialFeatures(degree=4) # N = 2   b2x^2 ' ye kadar git demek

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])

# regression model nesnemiz olan reg nesnemizi olusturup bunun fit metodunu cagırarak x_polynomial ve y eksenlerini fit ediyor
# yani regresyon modelimizi mevcut gercek verilerle egitiyoruz
reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])    # yapay zekayı egitme islemini burada yapıyoruz. Gerçek verilerle fit ediyoruz

#%% Artık modelimiz hazır ve egitilmis, simdi eldeki verilere göre modelimiz nasıl bir sonuc grafigi olusturuyor onu görelim
y_head = reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head,color="red",label="polynomial regression")
# plt.plot(xekseni,yekseni,color="green",label="linear regression")
plt.legend()

#veri setimizi de noktalı olarak scatter edelim de görelim uymus mu polynomial regression
plt.scatter(df['deneyim'],df['maas'])

plt.show()

# sonuca bakınca güzel uydugunu görüyoruz dogru regression modeli secmisiz. n degerlerini degistirerek fit deneylelim bakalım daha iyi sonuc verecekmi?
# n=4 yapınca tam oturdu. n'i sürekli arttırınca sistemi yavaslatır. o yüzden optimum degeri bulmak lazım.

x_polynomial1 = polynomial_regression.fit_transform([[4.5]])    # 4-5 arası dereceye uygun maaşı bulmaya calısıyoruz. # 2 paranteze alma sebebimiz 2 boyutlu bir array istiyor
reg.predict(x_polynomial1)











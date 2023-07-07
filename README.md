# Laporan Proyek Machine Learning - Bayu Abdurrosyid

## Domain Proyek

Domain proyek yang dipilih dalam proyek ini adalah bidang bisnis dengan judul proyek prediksi harga mobil bekas di UK.

## Latar Belakang

Pasar mobil bekas telah mengalami peningkatan pesat dari tahun ke tahun. Faktor ekonomi dan kemudahan menjadi alasan masyarakat lebih memilih mobil bekas sebagai alternatif yang lebih terjangkau dari membeli mobil baru. 
Pasar mobil bekas secara global bernilai sebesar USD 974,9 miliar pada tahun 2021 dan diperkirakan akan mencapai USD 1355,15 miliar pada tahun 2027 dengan tingkat pertumbuhan tahunan sebesar 7,80% dari tahun 2022 hingga 2027.
Banyaknya permintaan untuk mobil bekas menghasilkan kondisi pasar yang kompetitif. Kualitas mobil yang masih baik dan juga harga yang terjangkau merupakan hal utama yang dicari masyarakat dalam mencari mobil bekas. Namun menentukan harga yang wajar secara manual merupakan hal yang sulit karena banyak faktor seperti model mobil, ukuran mesin, dan usia dari mobil mempengaruhi harga jual dari mobil tersebut. 
Bagi penjual mobil bekas harga yang ditetapkan harus mencerminkan nilai rill dari mobil berdasarkan faktor faktor tertentu seperti usia jarak tempuh, ukuran mesin dan sebagainya. Harga yang terlalu tinggi dapat membuat mobil tidak laku terjual, sementara harga yang terlalu rendah dapat membuat penjual merugi. Bagi konsumen, prediksi harga mobil yang akurat dapat membantu konsumen memahami harga yang sesuai untuk dijadikan perbandingan saat membeli mobil bekas. Konsumen dapat mengerti apakah harga yang ditawarkan oleh penjual wajar atau tidak. 
Oleh karena itu, diperlukan suatu sistem yang dapat membantu memprediksi harga mobil yang akurat. Machine learning merupakan salah satu solusi yang dapat membantu prediksi dalam bisnis. Proyek  ini ditujukan untuk memprediksi harga mobil bekas menggunakan model machine learning. Diharapkan model yang dikembangkan mampu memprediksi harga mobil bekas yang sesuai agar dapat bersaing dipasar. Hasil prediksi tersebut dapat dijadikan acuan dasar bagi pemilik mobil untuk menyesuaikan harga mobilnya. 

## Business Understanding

### Problem Statements

Masalah yang ingin diselesaikan dalam penelitian ini adalah ketidakpastian dalam menentukan harga mobil bekas. Penentuan harga produk merupakan salah satu tahap yang krusial dalam bisnis. Untuk mengembangkan model yang dapat memprediksi harga mobil bekas diperlukan data latih yang baik kualitasnya. Beberapa hal perlu dipertimbangkan sebelum memberikan data ke model machine learning seperti keberadaan outlier, nilai kosong, serta fitur yang sesuai. Untuk itu deperlukan pra pemrosesan data mobil bekas yang baik untuk membuat model machine learning yang baik pula. Algoritma machine learning yang dipilih juga menentukan seberapa akurat model dapat memprediksi harga mobil bekas berdasarkan fitur-fitur tertentu.

### Goals

Tujuan dari proyek ini adalah mengembangkan model yang mampu memprediksi harga mobil bekas seakurat mungkin. Untuk mengembangkan model yang baik perlu diketaui fitur yang paling berpengaruh dalam menentukan harga mobil bekas serta perlu dilakukan persiapan data yang matang sebelum data harga mobil bekas dilatih ke model machine learning. 

### Solution statements

Berikut adalah beberapa solusi yang dapat diterapkan untuk menyelesaikan masalah dalam memprediksi harga mobil bekas
-	Memahami dan menganalisis fitur kategorik dan numerik pada data dengan melakukan visualisasi untuk mengetahui korelasi antar fitur, melakukan perbandingan antar fitur, dan mengetahui keberadaan outlier pada data.
-	Melakukan pra pemrosesan pada data menggunakan beberapa teknik seperti menghapus data outlier, mengganti nilai kosong pada data dengan nilai mean atau median
-	Mengembangkan model machine learning regresi untuk memprediksi hasil bilangan kontinyu. Algoritma machine learning yang dipakai antara lain K-Nearest Neighbor, Random Forest, dan AdaBoost

## Data Understanding

Dataset yang dipakai berjudul 100000 UK Used Car Dataset yang diunduh melalui [kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). Dataset ini terdiri dari 13 file berformat csv diantaranya audi.csv, bmw.csv, cclass.csv, focus.csv, ford.csv, hyundi.csv, merc.csv, skoda.csv, toyota.csv, unclean cclass.csv, unclean focus.csv, vauxhall.csv, dan vw.csv. File csv unclean cclass.csv dan unclean focus.csv tidak akan dipakai untuk melatih model, dikarenakan sudah terdapat data yang sama dan sudah dibersihakan pada file cclass.csv, dan focus.csv.

### Fitur-fitur yang terdapat pada dataset antara lain:

-	model : model atau brand dari mobil.
-	Year : tahun produksi mobil 
-	Price : harga dari mobil dalam mata uang euro
-	Transmission : jenis transmisi dari mobil
-	mileage : jarak yang sudah ditempuh oleh mobil selama pemakaian dalam mil
-	fuelType : jenis bahan bakar yang dipakai mobil
-	tax : jumlah pajak yang harus dibayar oleh pemilik mobil
-	mpg : jarak yang ditempuh (mil) per konsumsi bahan bakar (galon)
-	engineSize : ukuran volume dari mesin mobil



Kolom yang terdapat pada tiap file csv memilki jumlah dan nama yang sama, kecuali pada file hyundi.csv yang mana memiliki kolom bernama tax(£) yang tidak terdapat pada file lain.

![hyundai](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/hyundi.PNG?raw=true)

Sementara itu file lain memiliki kolom tax yang tidak dimiliki data file hyundi.csv, dan isi dari kolom tersebut sama-sama menyimpan nilai pajak dari mobil. Sehingga kolom tax(£) pada file hyundi.csv diganti nama menjadi tax. Setelah itu semua file csv digabung ke dalam satu dataframe.

Total seluruh file csv yang digabung memiliki jumlah data sebanyak 108540 baris. Dataset memiliki fitur numerik dan kategorik. Fitur numerik antara lain year, price, mileage, tax, mpg, engineSize. Sedangkan fitur kategorik antara lain model, transmission, dan fueltype. 

![describe](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/datadescribe.PNG?raw=true)

Selanjutnya akan dilakukan analisa pada masing-masing fitur terhadap kriteria jumlah data, rata-rata, standar deviasi, nilai minimum dan maksimum, serta nilai kuartil pada datase.
Dari tabel tersebut, dapat disimpulkan bahwa terdapat nilai 0 pada kolom tax dan engineSize yang tak merupakan hal yang tak wajar. Selain pada beberapa kolom seperti price, mileage, dan mpg, juga memiliki nilai yang terlalu besar dan terlalu kecil yang menandakan keberadaan outlier. Pada kolom tahun juga terdapat nilai tahun lebih dari 2023. Beberapa permasalahan tersebut akan diselesaikan pada pemaparan selanjutnya.

### Data Cleaning

#### Menangani nilai yang kosong
![zero](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/jh.PNG?raw=true)
Pada dataset, terdapat dua kolom yang memiliki nilai kosong yaitu kolom mpg dan tax. Saat dicek ternyata seluruh data yang tidak memiliki nilai mpg juga tidak memiliki nilai tax. Nilai kosong dapat diatasi dengan menghapus semua baris data yang kosong atau juga bisa dengan mengisi niliai kosong tersebut dengan nilai baru. Pada umumnya nilai yang dipakai untuk mengisi data kosong adalah nilai mean dan median. Nilai kosong pada kolom tax dan mileage diganti dengan nilai mean dari masing-masing fitur.
#### Menangani nilai 0

Dari tabel deskripsi dataset, terdapat beberapa kolom yang tak seharusnya memiliki nilai 0, yaitu tax dan juga engineSize. Tidak mungkin ada mesin mobil berukuran 0 
Nilai 0 tersebut akan diganti dengan nilai mean pada kolom tax dan median pada kolom engineSize.
#### Menangani Outlier

![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/box.png?raw=true)

Nilai outlier atau bisa disebut nilai pencilan, merupakan nilai ekstrim yang berbeda dari data yang lainnya. Keberadaan outlier pada dataset dapat berdampak buruk pada akurasi prediksi model, sehingga perlu dilakukan pengecekan outlier menggunakan grafik boxplot. Fitur-fitur numerik yang akan dicek antara lain price, mileage, tax, dan mpg.
Menangani nilai maksimal pada year
Pada kolom year terdapat data dengan nilai diatas 2023 yang tidak masuk akal. Maka dari itu data dengan tahun diatas 2023 akan disingkirkan.

### Univariative Analysis
Univariative analysis merupakan teknik analisis data yang menganalisa tiap fitur secara individu.

#### Analisa fitur kategorik
- Fitur model
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g1.png?raw=true)
Grafik diatas menunjukkan bahwa terdapat 156 jenis mobil yang terdapat pada dataset, dan mobil merk Fiesta merupakan yang paling banyak dipakai di UK dengan perbedaan banyak yang cukup signifikan dengan mobil kedua yaitu Focus.
- Fitur transmission
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g2.png?raw=true)
Grafik diatas menunjukkan bahwa terdapat 4 jenis transmisi yang terdapat pada dataset, yaitu manual, semi-auto, automatic, dan other. Transmisi manual merupakan yang paling populer. Sementara itu hanya terdapat 6 mobil yang memilki jenis transmisi berjenis other.
- Fitur fuelType
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g3.png?raw=true)
Grafik diatas menunjukkan bahwa terdapat 4 jenis bahan bakar yang terdapat pada dataset, yaitu petrol, diesel, hybrid, dan other. Bahan bakar petrol merupakan yang paling populer digunakan.

#### Analisa fitur numerik
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g4.png?raw=true)
Dari histogram diatas dapat disimpulkan beberapa hal diantaranya
-	Harga mobil memiliki rentang yang sangat tinggi yaitu dari rentang ribuan dolar, hingga kisaran 37000 dolar.
-	Mayoritas mobil berusia dibawah 10 tahun
-	Mileage atau Jarak tempuh mobil berbanding terbalik dengan jumlah data mobil

### Multivariative Analysis
Multivariative analysis merupakan teknik EDA yang menunjukkan hubungan antara dua atau lebih variabel pada data.  Pada dataset mobil bekas, akan dibandingkan beberapa fitur dengan fitur harga mobil.
#### Categorical Feature
Disini akan dilakukan perbandingan beberapa fitur dengan rata-rata harganya untuk mengetahui korelasi antara tiap fitur dengan fitur harga menggunakan grafik catplot.
![hyundai](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c1.png?raw=true)
Pada fitur model, rata-rata harga pada tiap jenis model mobil tidak memiliki pola tertentu. Tiap model mobil yang berbeda memiliki harga rata-rata yang masing-masing yang unik
![hyundai](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c2.png?raw=true)
Pada fitur transmisi, mobil dengan transmisi semi auto memiliki rata-rata harga tertinggi diatas 20000 euro. Sementara itu transmisi manual memilkl rata-rata harga terendah dikisaran 13000 euro.
![hyundai](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c3.png?raw=true)
Pada fitur fuelType, mobil berbahan bakar diesel memiliki harga rata-rata tertinggi, sementara itu mobil dengan bahan bakar petrol merupakan mobil dengan rata-rata harga terendah

#### Numerical Feature
Pada tahap berikut akan dilakukan perbandingan antar fitur numerik menggunakan pairplot dan juga correlation matrix.
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g5.png?raw=true)
Berdasarkan grafik pairplot diatas, dapat disimpukan beberapa hal berikut 
-	Terdapat beberapa fitur yang berkorelasi positif, diantaranya yaitu fitur price dengan year , fitur mpg dengan year, dan fitur price dengan engineSize.
-	Fitur yang berkorelasi negatif yaitu fitur price dengan mpg, mileage dengan year dan price dengan mileage.
-	Korelasi antar fitur yang lain tidak menunjukkan adanya pengaruh yang signifikan.

![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g6.png?raw=true)
Berdasarkan grafik correlation matrix diatas dapat disimpulkan bahwa
-	Fitur price memiliki korelasi positif yang sedang dengan fitur year (0.53) dan juga engineSize (0.6)
-	Fitur price memiliki korelasi negatif yang sedang dengan fitur mileage (-0.44) dan mpg (-0.37)
-	Fitur price memiliki korelasi paling lemah dengan fitur tax (0.18)
-	Fitur engineSize memiliki korelasi negatif yang kuat dengan fitur mileage (-0.75)
Dengan demikian berdasarkan grafik scatterplot dan correlation matrix, jika berpatokan pada fitur price dapat disimpulkan beberapa hal antara lain:
-	year: Mobil keluaran tahun terbaru cenderung memiliki harga yang lebih mahal dibanding mobil keluaran tahun yang lebih lama.
-	mileage: Mobil dengan mileage yang tinggi cenderung memiliki harga yang lebih murah. Mileage juga dipengauhi oleh umur mobil.
-	engineSize: Mobil dengan ukuran mesin yang besar cenderung memiliki harga yang lebih mahal dibanding mobil dengan ukuran mesin yang lebih kecil.
-	mpg: Mobil dengan mpg atau konsumsi bahan bakar mil per galon yang tinggi memiliki harga lebih rendah. Hal ini dapat disebabkan karena banyak mobil dengan harga yang lebih mahal memiliki ukuran mesin yang besar, sehingga mengkonsumsi lebih banyak bahan bakar.
-	tax: Jumlah pajak berbanding juga memiliki korelasi kecil dengan harga, dimana mobil dengan pajak yang tinggi cenderung memiliki harga yang tinggi

## Data Preparation
### Rekayasa fitur age
![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/age.png?raw=true)
Untuk mengecilkan interval dari fitur year, fitur year akan diganti dengan fitur age atau umur mobil yang didapat dari hasil 2023 dikurangi tahun pada baris.
### Label Encoding fitur kategorik
Selanjutnya akan dilakukan encoding pada fitur kategori. Dikarenakan pada fitur model terdapat banyak sekali jenis yang berbeda, maka encoding yang dipakai adalah labelencoder. Label encoder merupakan teknik encoding yang menerapkan encoding pada tiap jenis data kategorik dengan mengubah nilai kategorik menjadi nilai numerik mulai dari 0 hingga banyak kategori – 1.
### Train Test Split
Dataset dibagi ke bagian train dataset dan test dataset. Karena model yang dibuat akan memprediksi fitur price, maka kolom price akan menjadi target y dan sisanya menjadi fitur x.
### Standardization
Sebelum data diolah lebih lanjut, data akan distandardisasi menggunakan standard scaler sehingga memiliki standar deviasi 1 dan nilai rata-rata 0. Hal ini dilakukan karena model dapat bekerja lebih baik jika dilatih dengan data numerik yang memiliki skala yang relatif sama.

## Modeling
Tahap berikut adalah pengembangan model machine learning yang akan dipakai untuk memprediksi harga mobil bekas. Terdapat 3 algoritma yang akan dipakai antara lain k nearest neighbor, random forest, dan boosting algorithm.

- **K Nearest Neighbor**
K Nearest Neighbor bekerja dengan membandingkan jarak satu sampel dengan sampel lainnya pada grafik. Kemudian mengelompokkan sampel yang berdekatan dengan jumlah maksimal sampel tertentu pada satu kelompok. 
    -   Cara kerja algoritma K Nearest Neighbor
        -   Sampel nilai fitur akan dipilih, kemudian algoritma knn akan memillih sebanyak k sampel target yang memiliki nilai fitur terdekat terdekat dengan nilai yang fitur yang akan diprediksi.
        -   Semua nilai target dari sampel terdekat akan dirata-ratakan, menghasilkan nilai baru yang dapat dijadikan hasil prediksi
    -   Kekurangan
        -   Prediksi dilakukan cukup lambat dikarenakan knn harus mencari semua data terdekat terlebih dahulu dari data training sebelum melakukan prediksi
    -   Kelebihan
        -   KNN dapat menangani data yang non linear sescara baik.
        -   Fitting data pada model dilakukan relatif cepat

- **Random Forest**
Random forest seperti namanya merupakan algoritam yang menggunakan forest atau sekumpulan decision tree yang menentukan hasil prediksi. Random forest merupakan salah satu jenis model ensemble karena terdiri dari banyak algoritma decision tree yang saling mempengaruhi keputusan hasil prediksi yang pembagian data dan fiturnya dipilih secara acak.
    - Cara kerja algoritma Random Forest
        -   Memilih titik sampel acak pada data training
        -   Mengembangkan decision tree yang berasosiasi dengan titik sampel tersebut
Memilih jumlah n pohon yang akan dibangun, kemudian mengulangi tahap satu dan dua
        -   Untuk titk sampel yang baru, setiap n-pohon memprediksi nilai target dari sampel data, kemudian menambahkan nilai data baru tersebut pada rata-rata seluruh prediksi nilai target
    -   Kekurangan
        -   Overfitting masih dapat terjadi jika jumlah decision tree yang terlalu banyak
Random Forest memerlukan waktu yang cenderung lama untuk dapat belajar dari data
    -   Kelebihan
        -   Random forest memerlukan memori yang besar untuk menyimpan data banyak decision tree
        -   Random Forest dapat menangani data yang tidak teratur dan memiliki outlier. Random Forest dapat melakukan generalisasi dengan baik sehingga mengurangi kemungkinan overfitting
        -   Random Forest dapat melakukan prediksi dengan cepat
        -   Random Forest dapat mengukur seberapa penting fitur dari data sehingga dapat membantu seleksi fitur dan pemahaman data.

- **Boosting Algorithm**
Boosting algorithm bekerja dengan membangun model dari data latih, kemudian membuat model selanjutnya yang bertugas untuk memperbaiki dan meningkatkan akurasi dari model sebelumnya, begitu seterusnya hingga jumlah maksimum model yang ditentukan. Algoritma ini merupakan salah satu jenis ensemble algorithm yang terdiri dari beberapa model berupa decision tree yang saling bekerja sama. 
Boosting algorithm yang dipakai dalam penelitian ini adalah Adaptive Boosting yang diterapkan menggunakan AdaBoost.
    -   Cara kerja algoritma Adaptive Boosting
        -   Algoritma membangun decision tree sebagai model regresi lemah untuk setiap sampel data training, kemudian memberikan weight yang sama rata
        -   Model lemah yang memprediksi sampel data dengan selisih error yang tinggi akan diberikan weight lebih tinggi sehingga perannya lebih penting pada iterasi selanjutya
        -   Algorimta melakukan kalkulasi pada keseluruhan weight dari model, kemudian mengkombinasikan semua hasil prediksi dari model lemah dan berikan weight lebih tinggi pada model dengan performa baik
        -   Prediksi akhir didapat dari kombinasi weight, dimana model dengan weight yang lebih tinggi berkontribusi lebih besar pada hasil prediksi akhir
    -   Kekurangan
        -   Adaboost kurang cocok dengan data yang tidak beraturan atau tidak memiliki pola tertentu. 
    -   Kelebihan
        -   Adaptive Boosting memiliki kemungkinan yang rendah untuk terjadinya overfitting.


## Evaluasi
Evaluasi model dilakukan guna memeriksa tingkat akurasi yang dimilki oleh tiap model. Karena model machine learning merupakan model regresi yang mana memprediksi nilai kontinyu, maka metrik evaluasi yang dipakai adalah MSE atau Mean Squared Error. MSE menghitung jumlah selisih kuadrat rata-rata nilai yang sebenarnya dengan nilai hasil prediksi. Formula dari MSE adalah sebagai berikut

**Mean Squared Error**
MSE = $$\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$$ 

$$\left({n}\right)$$ = jumlah dataset = jumlah dataset
$${y}$$ = nilai sebenarnya
$$\hat{y}$$ = nilai prediksi

Berikut adalah tabel yang menampilkan hasil dari mean squared error pada ketiga model berdasarkan data training dan data testing

![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g6.5.PNG?raw=true)

Hasil evaluasi MSE tersebut juga dapat digambarkan ke dalam grafik batang seperti berikut


![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g7.png?raw=true)

Dari hasil evaluasi, dapat disimpulkan bahwa model yang menerapkan algoritma RandomForest merupakan model yang memiliki mse terkecil yang artinya hasil prediksi dari model tersebut merupakan yang paling mendekati nilai sebenarnya dibanding algoritma lainnya.

## Referensi
1. [Research and Markets. (2023, June 1). Global Used Car Market Report 2023: Sector is Expected to Reach $1,355 Billion by 2027 at a CAGR of 7.8%. GlobeNewswire. Retrieved July 1, 2023, from https://www.globenewswire.com/en/news-release/2023/06/01/2680613/28124/en/Global-Used-Car-Market-Report-2023-Sector-is-Expected-to-Reach-1-355-Billion-by-2027-at-a-CAGR-of-7-8.html ](https://www.globenewswire.com/en/news-release/2023/06/01/2680613/28124/en/Global-Used-Car-Market-Report-2023-Sector-is-Expected-to-Reach-1-355-Billion-by-2027-at-a-CAGR-of-7-8.html)
2. [Chen, J., Li, F., Xu, J., Wang, Q., Han, Q., & Yan, M. (2022, May). Comparisons of different methods used for second-hand car price prediction. In 2nd International Conference on Applied Mathematics, Modelling, and Intelligent Computing (CAMMIC 2022) (Vol. 12259, pp. 1191-1201). SPIE.](https://www.researchgate.net/publication/359504085_Comparisons_of_different_methods_used_for_second-hand_car_price_prediction)
3. [Miller, M. (2019, October 18). The Basics: KNN for classification and regression. Medium. Retrieved July 1, 2023, from https://towardsdatascience.com/the-basics-knn-for-classification-and-regression-c1e8a6c955 ](https://towardsdatascience.com/the-basics-knn-for-classification-and-regression-c1e8a6c955)
4. [Chaya. (2020, June 9). Random Forest Regression. Medium. Retrieved July 1, 2023, from https://levelup.gitconnected.com/random-forest-regression-209c0f354c84](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84)
5. [Wang, Y. (2023, February 19). What Are The Advantages And Disadvantages Of Random Forest?. Rebellion Research. Retrieved July 1, 2023, from https://www.rebellionresearch.com/what-are-the-advantages-and-disadvantages-of-random-forest#:~:text=In%20summary%2C%20Random%20Forest%20is,training%20time%2C%20and%20memory%20usage. ](https://www.rebellionresearch.com/what-are-the-advantages-and-disadvantages-of-random-forest#:~:text=In%20summary%2C%20Random%20Forest%20is,training%20time%2C%20and%20memory%20usage.)
6. [Chengsheng, T., Huacheng, L., & Bing, X. (2017). AdaBoost typical Algorithm and its application research. In MATEC Web of Conferences (Vol. 139, p. 00222). EDP Sciences.](https://www.researchgate.net/publication/321583409_AdaBoost_typical_Algorithm_and_its_application_research/fulltext/5a29be04a6fdccfbbf8185df/AdaBoost-typical-Algorithm-and-its-application-research.pdf)

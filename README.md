# Laporan Proyek Machine Learning - Bayu Abdurrosyid

## Domain Proyek

Domain proyek yang dipilih dalam proyek ini adalah bidang bisnis dengan judul proyek prediksi harga mobil bekas di UK.

## Latar Belakang

Pasar mobil bekas telah mengalami peningkatan pesat dari tahun ke tahun. Faktor ekonomi dan kemudahan menjadi alasan masyarakat lebih memilih mobil bekas sebagai alternatif yang lebih terjangkau dari membeli mobil baru. 
Pasar mobil bekas secara global bernilai sebesar 974,9 miliar USD pada tahun 2021 dan diperkirakan akan mencapai 1355,15 miliar USD pada tahun 2027 dengan tingkat pertumbuhan tahunan sebesar 7,80% dari tahun 2022 hingga 2027.
Banyaknya permintaan untuk mobil bekas menghasilkan kondisi pasar yang kompetitif. Kualitas mobil yang masih baik dan juga harga yang terjangkau merupakan hal utama yang dicari masyarakat dalam mencari mobil bekas. Namun menentukan harga yang wajar secara manual merupakan hal yang sulit karena banyak faktor seperti model mobil, ukuran mesin, dan usia dari mobil mempengaruhi harga jual dari mobil tersebut. 
Bagi penjual mobil bekas harga yang ditetapkan harus mencerminkan nilai riil dari mobil berdasarkan faktor faktor tertentu seperti usia jarak tempuh, ukuran mesin dan sebagainya. Harga yang terlalu tinggi dapat membuat mobil tidak laku terjual, sementara harga yang terlalu rendah dapat membuat penjual merugi. Bagi konsumen, prediksi harga mobil yang akurat dapat membantu konsumen memahami harga yang sesuai untuk dijadikan perbandingan saat membeli mobil bekas. Konsumen dapat mengerti apakah harga yang ditawarkan oleh penjual wajar atau tidak. 
Oleh karena itu, diperlukan suatu sistem yang dapat membantu memprediksi harga mobil yang akurat. _Machine learning_ merupakan salah satu solusi yang dapat membantu prediksi dalam bisnis. Proyek  ini ditujukan untuk memprediksi harga mobil bekas menggunakan model _machine learning_. Diharapkan model yang dikembangkan mampu memprediksi harga mobil bekas yang sesuai agar dapat bersaing dipasar. Hasil prediksi tersebut dapat dijadikan acuan dasar bagi pemilik mobil untuk menyesuaikan harga mobilnya. 

## Business Understanding

### Problem Statements

Berdasarkan latar belakang diatas, masalah yang ingin diselesaikan dalam penelitian ini antara lain:
- Bagaimana cara menyelesaikan masalah dalam penentuan harga mobil bekas yang tidak pasti?
- Bagaimana korelasi harga mobil dengan fitur-fitur yang terdapat pada dataset?
- Bagaimana cara melakukan pra pemrosesan data sebelum data dilatih ke model _machine learning_?
- Bagaimana cara melatih model yang baik untuk memprediksi harga mobil bekas?
- Algoritma apa yang memiliki tingkat galat terendah dalam memprediksi harga mobil bekas berdasarkan _dataset_?

### Goals

- Mencari solusi untuk akan permasalahan dalam memprediksi harga mobil yang wajar dengan lebih akurat.
- Mengidentifikasi fitur-fitur yang berpengaruh secara signifikan terhadap penentuan harga mobil.
- Melakukan pra pemrosesan data pada _dataset_ dengan tepat agar dapat melatih model dengan baik.
- Mengembangkan model _machine learning_ yang dapat memprediksi harga mobil bekas.
- Mengevaluasi model _machine learning_ untuk mencari model yang memiliki tingkat galat paling rendah.

### Solution statements

Berikut adalah beberapa solusi yang dapat diterapkan untuk menyelesaikan masalah dalam memprediksi harga mobil bekas
-	Memahami dan menganalisis fitur kategorik dan numerik pada data dengan melakukan visualisasi untuk mengetahui korelasi antar fitur, melakukan perbandingan antar fitur, dan mengetahui keberadaan outlier pada data.
-	Melakukan pra pemrosesan pada data menggunakan beberapa teknik seperti menghapus data outlier, mengganti nilai kosong pada data dengan nilai mean atau median
-	Mengembangkan model _machine learning_ regresi untuk memprediksi hasil bilangan kontinyu. Algoritma _machine learning_ yang dipakai antara lain _K-Nearest Neighbor_, _Random Forest_, dan _AdaBoost_

## Data Understanding

_Dataset_ yang dipakai berjudul _100000 UK Used Car Dataset_ yang diunduh melalui [kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). 

Berikut adalah pemaparan lanjut mengenai informasi pada _dataset_:

- _Dataset_ terdiri dari 13 file berformat csv diantaranya audi.csv, bmw.csv, cclass.csv, focus.csv, ford.csv, hyundi.csv, merc.csv, skoda.csv, toyota.csv, unclean cclass.csv, unclean focus.csv, vauxhall.csv, dan vw.csv. Setiap file memiliki jumlah data dan kolom yang berbeda. Jumlah baris dan kolom dari setiap file dirincikan pada tabel 1 berikut.

Tabel 1. Rincian _dataset_
|nama file|baris|kolom numerik|kolom kategorik| ukuran file (kb)|
|-|-|-|-|-|
|audi.csv| 10668 |6|3|521.55|
|bmw.csv| 10781 |6|3|578.83|
|cclass.csv| 3899 |4|3|180.52|
|focus.csv |5454 |4|3|228.64|
|ford.csv |17965 |6|3|911.42|
|hyundi.csv| 4860 |6|3|242.43|
|merc.csv |13119 |6|3|723.80|
|skoda.csv |6267 |6|3|327.28|
|toyota.csv |6738 |6|3|339.71|
|unclean cclass.csv| 4006 |1|10|317.46|
|unclean focus.csv| 5604 |1|10|414.51|
|vauxhall.csv |13632 |6|3|698.63|
|vw.csv |15157 |6|3|769.66|

Sebagian besar file memiliki jumlah kolom yang sama kecuali pada file cclass, focus, unclean cclass, dan unclean focus. Pada file cclass dan juga focus tidak terdapat kolom tax dan mpg. Masalah ini dapat diatasi dengan mengganti nilai kosong tersebut dengan nilai baru.
Sementara itu pada file unclean cclass dan unclean focus terdapat beberapa kolom yang tidak ada pada file lain. Namun karena data pada unclean cclass dan juga cclass cenderung sama begitu pula data pada file unclean focus dan focus, maka file unclean cclass dan unclean focus dapat diabaikan.
Data dari file hyundi.csv juga sedikit berbeda dengan data yang lainnya.Pada file hyundi fitur pajak diberi nama tax(£) sementara pada file lain fitur pajak diberi nama tax. Untuk itu sebelum seluruh data digabung, data hyundi akan diolah terlebih dahulu dengan mengubah nama fitur tax(£).

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

Tabel 2. Data hyundi.csv
|model|year|price|transmission|mileage|fuelType|tax(£)|mpg|engineSize|
|-|-|-|-|-|-|-|-|-|
|I20| 2017| 7999| Manual| 17307| Petrol| 145| 58.9| 1.2|
|Tucson| 2016| 14499| Automatic| 25233| Diesel| 235| 43.5| 2.0|
|Tucson |2016 |11399 |Manual |37877 |Diesel |30 |61.7 |1.7|
|I19| 2016| 6499| Manual| 23789| Petrol| 20 |60.1| 1.0|
|IX35 |2015 |10199 |Manual |33177 |Diesel| 160 |51.4 |2.0|

Sementara itu file lain memiliki kolom tax yang tidak dimiliki data file hyundi.csv, dan isi dari kolom tersebut sama-sama menyimpan nilai pajak dari mobil. Sehingga kolom tax(£) pada file hyundi.csv diganti nama menjadi tax. Setelah itu semua file csv digabung ke dalam satu dataframe. Total seluruh file csv yang digabung memiliki jumlah data sebanyak 108540. _Dataset_ memiliki fitur numerik dan kategorik.

Tabel 3. Deskripsi _dataset_
|  |year|	price|	mileage|	tax|	mpg|	engineSize|
|-|-|-|-|-|-|-|
|count|	108540.000000|	108540.000000|	108540.000000|	99187.000000|	99187.000000|	108540.000000|
|mean|	2017.098028|	16890.124046|	23025.928469|	120.299838|	55.166825|	1.661644|
|std|	2.130057|	9756.266820|	21176.423684|	63.150926|	16.138522|	0.557058|
|min|	1970.000000|	450.000000|	1.000000|	0.000000|	0.300000|	0.000000|
|25%	|2016.000000|	10229.500000|	7491.750000|	125.000000|	47.100000|	1.200000|
|50%|	2017.000000|	14698.000000|	17265.000000|	145.000000|	54.300000|	1.600000|
|75%	|2019.000000|	20940.000000|	32236.000000|	145.000000|	62.800000|	2.000000|
|max|	2060.000000|	159999.000000|	323000.000000|	580.000000|	470.800000|	6.600000|

Selanjutnya akan dilakukan analisis pada masing-masing fitur terhadap kriteria jumlah data, rata-rata, standar deviasi, nilai minimum dan maksimum, serta nilai kuartil pada dataset.
Dari tabel tersebut, dapat disimpulkan bahwa terdapat nilai 0 pada kolom tax dan engineSize yang tak merupakan hal yang tak wajar. Selain pada beberapa kolom seperti price, mileage, dan mpg, juga memiliki nilai yang terlalu besar dan terlalu kecil yang menandakan keberadaan _outlier_ atau data pencilan. Pada kolom tahun juga terdapat nilai tahun lebih dari 2023. Beberapa permasalahan tersebut akan diselesaikan pada pemaparan selanjutnya.

### Data Cleaning

#### Menangani nilai yang kosong

Tabel 4. Nilai kosong
|fitur|nilai kosong|
|-|-|
|model|              0|
|year|               0|
|price|              0|
|transmission|       0|
|mileage|            0|
|fuelType|           0|
|tax|             9353|
|mpg|             9353|
|engineSize|         0|

Pada _dataset_, terdapat dua kolom yang memiliki nilai kosong yaitu kolom mpg dan tax. Saat dicek ternyata seluruh data yang tidak memiliki nilai mpg juga tidak memiliki nilai tax. Nilai kosong dapat diatasi dengan menghapus semua baris data yang kosong atau juga bisa dengan mengisi nilai kosong tersebut dengan nilai baru. Pada umumnya nilai yang dipakai untuk mengisi data kosong adalah nilai _mean_ dan _median_. Nilai kosong pada kolom tax dan mileage diganti dengan nilai _mean_ dari masing-masing fitur.
#### Menangani nilai 0

Seperti yang tertera pada tabel 3 mengenai deskripsi _dataset_, terdapat beberapa kolom yang tak seharusnya memiliki nilai 0, yaitu tax dan juga engineSize. Tidak mungkin ada mesin mobil berukuran 0.
Nilai 0 tersebut akan diganti dengan nilai _mean_ pada kolom tax dan _median_ pada kolom engineSize.
#### Menangani Outlier

![boxplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/box.png?raw=true)
Gambar 1. _Boxplot_ fitur numerik

Nilai _outlier_ atau bisa disebut nilai pencilan, merupakan nilai ekstrim yang berbeda dari data yang lainnya. Keberadaan _outlier_ pada dataset dapat berdampak buruk pada akurasi prediksi model, sehingga perlu dilakukan pengecekan _outlier_ menggunakan grafik _boxplot_. Fitur-fitur numerik yang akan dicek antara lain price, mileage, tax, dan mpg.
Menangani nilai maksimal pada year
Pada kolom year terdapat data dengan nilai diatas 2023 yang tidak masuk akal. Maka dari itu data dengan tahun diatas 2023 akan disingkirkan.

### Univariative Analysis
_Univariative analysis_ merupakan teknik analisis data yang menganalisis tiap fitur secara individu.

#### Analisis fitur kategorik
- Fitur model
![model](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g1.png?raw=true)
Gambar 2. Frekuensi fitur model

    Grafik diatas menunjukkan bahwa terdapat 156 jenis mobil yang terdapat pada _dataset_, dan mobil model Fiesta merupakan yang paling banyak dipakai di UK dengan perbedaan banyak yang cukup signifikan dengan mobil kedua yaitu Focus.
- Fitur transmission
![transmission](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g2.png?raw=true)
Gambar 3. Frekuensi fitur transmission

    Grafik diatas menunjukkan bahwa terdapat 4 jenis transmisi yang terdapat pada _dataset_, yaitu _manual_, _semi-auto_, _automatic_, dan _other_. Transmisi _manual_ merupakan yang paling populer. Sementara itu hanya terdapat 6 mobil yang memilki jenis transmisi berjenis _other_.
- Fitur fuelType
![fuelType](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g3.png?raw=true)
Gambar 4. Frekuensi fitur fuelType

    Grafik diatas menunjukkan bahwa terdapat 4 jenis bahan bakar yang terdapat pada dataset, yaitu _petrol_, _diesel_, _hybrid_, dan _other_. Bahan bakar _petrol_ merupakan yang paling populer digunakan.

#### Analisis fitur numerik
![histogram](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g4.png?raw=true)
Dari histogram diatas dapat disimpulkan beberapa hal diantaranya
Gambar 5. Histogram fitur numerik

-	Harga mobil memiliki rentang yang sangat tinggi yaitu dari rentang ribuan euro, hingga kisaran 37000 euro.
-	Mayoritas mobil berusia dibawah 10 tahun
-	Mileage atau Jarak tempuh mobil berbanding terbalik dengan jumlah data mobil

### Multivariative Analysis
_Multivariative analysis_ merupakan teknik _EDA_ yang menunjukkan hubungan antara dua atau lebih variabel pada data.  Pada dataset mobil bekas, akan dibandingkan beberapa fitur dengan fitur harga mobil.
#### Categorical Feature
Disini akan dilakukan perbandingan beberapa fitur dengan rata-rata harganya untuk mengetahui korelasi antara tiap fitur dengan fitur harga menggunakan grafik _catplot_.
![model](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c1.png?raw=true)
Gambar 6. Grafik price relatif terhadap model

Pada fitur model, rata-rata harga pada tiap jenis model mobil tidak memiliki pola tertentu. Tiap model mobil yang berbeda memiliki harga rata-rata yang masing-masing yang unik.
![transmission](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c2.png?raw=true)
Gambar 7. Grafik price relatif terhadap transmission

Pada fitur transmisi, mobil dengan transmisi _semi auto_ memiliki rata-rata harga tertinggi diatas 20000 euro. Sementara itu transmisi _manual_ memiliki rata-rata harga paling rendah yang berkisar diangka 13000 euro.
![fuelType](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/c3.png?raw=true)
Gambar 8. Grafik price relatif terhadap fuelType

Pada fitur fuelType, mobil berbahan bakar _diesel_ memiliki harga rata-rata tertinggi, sementara itu mobil dengan bahan bakar _petrol_ merupakan mobil dengan rata-rata harga terendah

#### Numerical Feature
Pada tahap berikut akan dilakukan perbandingan antar fitur numerik menggunakan _pairplot_ dan juga _correlation matrix_.
![pairplot](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g5.png?raw=true)
Gambar 9. _Pairplot_ fitur numerik

Berdasarkan grafik _pairplot_ diatas, dapat disimpukan beberapa hal berikut 
-	Terdapat beberapa fitur yang berkorelasi positif, diantaranya yaitu fitur price dengan year , fitur mpg dengan year, dan fitur price dengan engineSize.
-	Fitur yang berkorelasi negatif yaitu fitur price dengan mpg, mileage dengan year dan price dengan mileage.
-	Korelasi antar fitur yang lain tidak menunjukkan adanya pengaruh yang signifikan.

![cor matrix](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g6.png?raw=true)
Gambar 10. _Correlation matrix_ fitur numerik

Berdasarkan grafik correlation matrix diatas dapat disimpulkan bahwa
-	Fitur price memiliki korelasi positif yang sedang dengan fitur year (0.53) dan juga engineSize (0.6)
-	Fitur price memiliki korelasi negatif yang sedang dengan fitur mileage (-0.44) dan mpg (-0.37)
-	Fitur price memiliki korelasi paling lemah dengan fitur tax (0.18)
-	Fitur engineSize memiliki korelasi negatif yang kuat dengan fitur mileage (-0.75)

Dengan demikian berdasarkan grafik _scatterplot_ dan _correlation matrix_, jika berpatokan pada fitur price dapat disimpulkan beberapa hal antara lain:
-	year: Mobil keluaran tahun terbaru cenderung memiliki harga yang lebih mahal dibanding mobil keluaran tahun yang lebih lama.
-	mileage: Mobil dengan mileage yang tinggi cenderung memiliki harga yang lebih murah. Mileage juga dipengauhi oleh umur mobil.
-	engineSize: Mobil dengan ukuran mesin yang besar cenderung memiliki harga yang lebih mahal dibanding mobil dengan ukuran mesin yang lebih kecil.
-	mpg: Mobil dengan mpg atau konsumsi bahan bakar mil per galon yang tinggi memiliki harga lebih rendah. Hal ini dapat disebabkan karena banyak mobil dengan harga yang lebih mahal memiliki ukuran mesin yang besar, sehingga mengkonsumsi lebih banyak bahan bakar.
-	tax: Jumlah pajak berbanding juga memiliki korelasi kecil dengan harga, dimana mobil dengan pajak yang tinggi cenderung memiliki harga yang tinggi

## Data Preparation
### Rekayasa fitur age

Untuk mengurangi interval dari fitur year, fitur year akan diganti dengan fitur age atau umur mobil yang didapat dari hasil 2023 dikurangi tahun pada baris. Sehingga hasilnya tampak seperti tabel 5 berikut.

Tabel 5. Data dengan kolom age
|	model|	price|	transmission|	mileage|	fuelType|	tax|	mpg|	engineSize|	age|
|-|-|-|-|-|-|-|-|-|
|	T-Roc|	25000|	Automatic|	13904|	Diesel|	145.0|	49.6|	2.0|	4|
|	T-Roc|	26883|	Automatic|	4562|	Diesel|	145.0|	49.6|	2.0|	4|
|	T-Roc|	20000|	Manual|	7414|	Diesel|	145.0|	50.4|	2.0|	4|
|	T-Roc|	33492|	Automatic|	4825|	Petrol|	145.0|	32.5|	2.0|	4|
|	T-Roc|	22900|	Semi-Auto|	6500|	Petrol|	150.0|	39.8|	1.5|	4|

### Label Encoding fitur kategorik
Selanjutnya akan dilakukan _encoding_ pada fitur kategori. Dikarenakan pada fitur model terdapat banyak sekali jenis yang berbeda, maka _encoding_ yang dipakai adalah _label encoder_. _Label encoder_ merupakan teknik _encoding_ yang menerapkan _encoding_ pada tiap jenis data kategorik dengan mengubah nilai kategorik menjadi nilai numerik mulai dari 0 hingga banyak kategori – 1.
### Train Test Split
_Dataset_ dibagi ke bagian _train dataset_ dan _test dataset_. Karena model yang dibuat akan memprediksi fitur price, maka kolom price akan menjadi target y dan sisanya menjadi fitur x.
### Standardization
Sebelum data diolah lebih lanjut, data akan distandardisasi menggunakan _standard scaler_ sehingga memiliki standar deviasi 1 dan nilai rata-rata 0. Hal ini dilakukan karena model dapat bekerja lebih baik jika dilatih dengan data numerik yang memiliki skala yang relatif sama.

## Modeling
Tahap berikut adalah pengembangan model machine learning yang akan dipakai untuk memprediksi harga mobil bekas. Terdapat 3 algoritma yang akan dipakai antara lain _k nearest neighbor_, _random forest_, dan _boosting algorithm_.

- **K Nearest Neighbor**
_K Nearest Neighbor_ bekerja dengan membandingkan jarak satu sampel dengan sampel lainnya pada grafik. Kemudian mengelompokkan sampel yang berdekatan dengan jumlah maksimal sampel tertentu pada satu kelompok. 
    -   Cara kerja algoritma _K Nearest Neighbor_
        -   Sampel nilai fitur akan dipilih, kemudian algoritma knn akan memillih sebanyak k sampel target yang memiliki nilai fitur terdekat terdekat dengan nilai yang fitur yang akan diprediksi.
        -   Semua nilai target dari sampel terdekat akan dirata-ratakan, menghasilkan nilai baru yang dapat dijadikan hasil prediksi
    -   Kekurangan
        -   Prediksi dilakukan cukup lambat dikarenakan knn harus mencari semua data terdekat terlebih dahulu dari data training sebelum melakukan prediksi
    -   Kelebihan
        -   _KNN_ dapat menangani data yang non linear dengan baik.
        -   Fitting data pada model dilakukan relatif cepat
        
    Proyek ini menggunakan method sklearn.neighbors.KNeighborsRegressor untuk membuat model regresi _KNN_. Adapun parameter yang dipakai antara lain:
    - n_neighbors = jumlah tetangga terdekat maksimal yang diberi nilai 10

- **Random Forest**
_Random forest_ seperti namanya merupakan algoritma yang menggunakan _forest_ atau sekumpulan _decision tree_ yang menentukan hasil prediksi. _Random forest_ merupakan salah satu jenis model _ensemble_ karena terdiri dari banyak algoritma _decision tree_ yang saling mempengaruhi keputusan hasil prediksi yang pembagian data dan fiturnya dipilih secara acak.
    - Cara kerja algoritma _Random Forest_
        -   Memilih titik sampel acak pada data training
        -   Mengembangkan _decision tree_ yang berasosiasi dengan titik sampel tersebut
Memilih jumlah n pohon yang akan dibangun, kemudian mengulangi tahap satu dan dua
        -   Untuk titik sampel yang baru, setiap n-pohon memprediksi nilai target dari sampel data, kemudian menambahkan nilai data baru tersebut pada rata-rata seluruh prediksi nilai target
    -   Kekurangan
        -   _Overfitting_ masih dapat terjadi jika jumlah decision tree yang terlalu banyak
Random Forest memerlukan waktu yang cenderung lama untuk dapat belajar dari data
    -   Kelebihan
        -   _Random forest_ memerlukan memori yang besar untuk menyimpan data banyak decision tree
        -   _Random Forest_ dapat menangani data yang tidak teratur dan memiliki outlier. Random Forest dapat melakukan generalisasi dengan baik sehingga mengurangi kemungkinan overfitting
        -   _Random Forest_ dapat melakukan prediksi dengan cepat
        -   _Random Forest_ dapat mengukur seberapa penting fitur dari data sehingga dapat membantu seleksi fitur dan pemahaman data.
        
    Proyek ini menggunakan method sklearn.ensemble.RandomForestRegressor untuk membuat model regresi _Random Forest_. Adapun parameter yang dipakai antara lain:
    - n_estimators = jumlah _estimator decision tree_ maksimal yang diberi nilai 50
    - max_depth = jumlah kedalaman _decision tree_ maksimal yang diberi nilai 16
    - random_state = _seed random_ agar dapat menghasilkan model yang sama persis jika model dilatih ulang yang diberi nilai 55
    - n_jobs = jumlah proses yang dijalankan secara pararel yang diberi nilai -1 sehingga proses menggunakan semua prosesor
    
- **Boosting Algorithm**
_Boosting algorithm_ bekerja dengan membangun model dari data latih, kemudian membuat model selanjutnya yang bertugas untuk memperbaiki dan meningkatkan akurasi dari model sebelumnya, begitu seterusnya hingga jumlah maksimal model yang ditentukan. Algoritma ini merupakan salah satu jenis _ensemble algorithm_ yang terdiri dari beberapa model berupa _decision tree_ atau bisa disebut _weak learner_ yang saling bekerja sama dan meningkatkan performa di setiap iterasi. 
_Boosting algorithm_ yang dipakai dalam penelitian ini adalah _Adaptive Boosting_ yang diterapkan menggunakan _AdaBoost_.
    -   Cara kerja algoritma _Adaptive Boosting_
        -   Algoritma membangun _decision tree_ sebagai model regresi lemah untuk setiap sampel data training, kemudian memberikan _weight_ yang sama rata
        -   Model lemah yang memprediksi sampel data dengan selisih _error_ yang tinggi akan diberikan _weight_ lebih tinggi sehingga perannya lebih penting pada iterasi selanjutnya
        -   Algoritma melakukan kalkulasi pada keseluruhan _weight_ dari model, kemudian mengombinasikan semua hasil prediksi dari model lemah dan berikan _weight_ lebih tinggi pada model dengan performa baik
        -   Prediksi akhir didapat dari kombinasi _weight_, dimana model dengan _weight_ yang lebih tinggi berkontribusi lebih besar pada hasil prediksi akhir
    -   Kekurangan
        -   _Adaboost_ kurang cocok dengan data yang tidak beraturan atau tidak memiliki pola tertentu. 
    -   Kelebihan
        -   _Adaptive Boosting_ memiliki kemungkinan yang rendah untuk terjadinya _overfitting_.

    Proyek ini menggunakan method sklearn.ensemble.AdaBoostRegressor untuk membuat model regresi _Adaptive Boosting_. Adapun parameter yang dipakai antara lain:
    - learning_rate = _weight_ yang diterapkan pada setiap model pada setiap iterasi boosting yang diberi nilai 0.5 
    - random_state = _seed random_ agar dapat menghasilkan model yang sama persis jika model dilatih ulang yang diberi nilai 55


## Evaluasi
Evaluasi model dilakukan guna memeriksa tingkat akurasi yang dimiliki oleh tiap model. Karena model _machine learning_ yang telah dikembangkan berjenis regresi yang mana memprediksi nilai prediksi berupa angka kontinyu, maka metrik evaluasi yang dipakai adalah _MSE_ atau _Mean Squared Error_. _MSE_ menghitung jumlah selisih kuadrat rata-rata nilai yang sebenarnya dengan nilai hasil prediksi. Formula dari _MSE_ adalah sebagai berikut

**Mean Squared Error**
$$MSE = \frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$$ 

n = jumlah data pada _dataset_

y = nilai sebenarnya

ŷ = nilai prediksi

Berikut adalah tabel yang menampilkan hasil dari _mean squared error_ pada ketiga model berdasarkan data _training_ dan data _testing_

Tabel 6. _MSE_ setiap algoritma
| |	train|	test|
|-|-|-|
|KNN|	2157.043542|	2686.927433|
|RF|	1244.647867|	2468.617218|
|Boosting|	16145.772919|	15677.008501|

Hasil evaluasi _MSE_ tersebut juga dapat digambarkan ke dalam grafik batang pada gambar 11 seperti berikut.

![graph](https://github.com/TehBotolBayu/Predictive-Analysis/blob/main/g7.png?raw=true)
Gambar 11.Grafik MSE setiap algoritma

Dari hasil evaluasi, dapat disimpulkan bahwa model yang menerapkan algoritma _Random Forest_ merupakan model yang memiliki nilai _MSE_ terkecil yang artinya hasil prediksi dari model tersebut merupakan yang paling mendekati nilai sebenarnya dibanding algoritma lainnya. Hasil prediksi dari setiap algoritma terhadap sampel data _test_ dirincikan pada tabel  7 berikut.

Tabel 7. Hasil prediksi
|	|y_true|	prediksi_KNN|	prediksi_RF|	prediksi_Boosting|
|-|-|-|-|-|
|5654|	24000|	26716.0|	24015.3|	24251.1|
|1131|	22052|	22618.5|	22737.5|	20145.6|
|707|	18995|	18464.6|	17452.2|	19628.3|
|5003|	28655|	28557.0|	28541.4|	27764.1|
|2247|	12995|	20504.2|	16462.2|	12457.6|

Dari 5 sampel data, aloritma _KNN_ menghasilkan 3 prediksi yang nilainya paling mendekati nilai sebenarnya dibanding kedua algoritma lainnya. Sementara itu algoritma _Random Forest_ dan _AdaBoost_ sama-sama berhasil menghasilkan 1 nilai prediksi yang paling mendekati nilai sebenarnya.
Secara keseluruhan hasil prediksi menunjukkan bahwa algoritma _KNN_ dan _Random Forest_ menghasilkan prediksi yang paling mendekati nilai sebenarnya pada ke 5 sampel data. Sedangkan algoritma _AdaBoost_ menghasilkan hasil prediksi yang jauh dari nilai asli dibandingkan kedua algoritma lainnya.

## Kesimpulan
Model yang dikembangkan telah dapat memprediksi harga mobil bekas dengan tingkat akurasi yang cukup tinggi dengan mempelajari _dataset_ yang sudah diproses terlebih dahulu. Melalui evaluasi, didapat algoritma _Random Forest_ merupakan algorima yang memiliki tingkat _error_ prediksi terendah. Diharapkan model yang sudah dikembangkan dapat membantu pelaku bisnis maupun calon konsumen untuk memprediksi harga mobil bekas secara otomatis dengan lebih akurat.

## Referensi
[1] [Research and Markets. (2023, June 1). Global Used Car Market Report 2023: Sector is Expected to Reach $1,355 Billion by 2027 at a CAGR of 7.8%. GlobeNewswire. Retrieved July 1, 2023, from https://www.globenewswire.com/en/news-release/2023/06/01/2680613/28124/en/Global-Used-Car-Market-Report-2023-Sector-is-Expected-to-Reach-1-355-Billion-by-2027-at-a-CAGR-of-7-8.html ](https://www.globenewswire.com/en/news-release/2023/06/01/2680613/28124/en/Global-Used-Car-Market-Report-2023-Sector-is-Expected-to-Reach-1-355-Billion-by-2027-at-a-CAGR-of-7-8.html)
[2] [Chen, J., Li, F., Xu, J., Wang, Q., Han, Q., & Yan, M. (2022, May). Comparisons of different methods used for second-hand car price prediction. In 2nd International Conference on Applied Mathematics, Modelling, and Intelligent Computing (CAMMIC 2022) (Vol. 12259, pp. 1191-1201). SPIE.](https://www.researchgate.net/publication/359504085_Comparisons_of_different_methods_used_for_second-hand_car_price_prediction)
[3] [Miller, M. (2019, October 18). The Basics: KNN for classification and regression. Medium. Retrieved July 1, 2023, from https://towardsdatascience.com/the-basics-knn-for-classification-and-regression-c1e8a6c955 ](https://towardsdatascience.com/the-basics-knn-for-classification-and-regression-c1e8a6c955)
[4] [Chaya. (2020, June 9). Random Forest Regression. Medium. Retrieved July 1, 2023, from https://levelup.gitconnected.com/random-forest-regression-209c0f354c84](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84)
[5] [Wang, Y. (2023, February 19). What Are The Advantages And Disadvantages Of Random Forest?. Rebellion Research. Retrieved July 1, 2023, from https://www.rebellionresearch.com/what-are-the-advantages-and-disadvantages-of-random-forest#:~:text=In%20summary%2C%20Random%20Forest%20is,training%20time%2C%20and%20memory%20usage. ](https://www.rebellionresearch.com/what-are-the-advantages-and-disadvantages-of-random-forest#:~:text=In%20summary%2C%20Random%20Forest%20is,training%20time%2C%20and%20memory%20usage.)
[6] [Chengsheng, T., Huacheng, L., & Bing, X. (2017). AdaBoost typical Algorithm and its application research. In MATEC Web of Conferences (Vol. 139, p. 00222). EDP Sciences.](https://www.researchgate.net/publication/321583409_AdaBoost_typical_Algorithm_and_its_application_research/fulltext/5a29be04a6fdccfbbf8185df/AdaBoost-typical-Algorithm-and-its-application-research.pdf)

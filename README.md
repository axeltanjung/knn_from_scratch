# K-Nearest Neighbor From Scratch

## Latar Belakang
Dalam era digital saat ini, jumlah data yang dihasilkan dan dikumpulkan terus meningkat secara eksponensial. Data-data tersebut menjadi aset berharga bagi berbagai industri dan sektor, seperti bisnis, kesehatan, keuangan, dan lain-lain. Namun, data yang berlimpah ini hanya memiliki nilai yang signifikan jika dapat dianalisis dan dimanfaatkan secara efektif untuk mengambil keputusan yang tepat. Di sinilah peran dari algoritma machine learning menjadi sangat penting.

Salah satu algoritma machine learning yang sederhana namun efektif adalah k-Nearest Neighbors (k-NN). Algoritma ini termasuk dalam kelompok algoritma supervised learning dan digunakan dalam berbagai kasus, seperti klasifikasi dan regresi. K-NN memiliki konsep dasar yang mudah dipahami, yaitu mengklasifikasikan atau memprediksi suatu data baru berdasarkan mayoritas kategori data terdekatnya. Meskipun sederhana, k-NN memiliki potensi yang kuat dalam mengatasi berbagai permasalahan data mining dan analisis.

Namun, untuk memahami sepenuhnya bagaimana k-NN bekerja dan menerapkannya dengan benar, sangat penting untuk memiliki pemahaman yang mendalam tentang konsep di balik algoritma ini. Mengimplementasikan k-NN dari awal membantu para praktisi dan peneliti dalam memahami inti dari algoritma ini, bagaimana proses seleksi tetangga terdekat dilakukan, dan bagaimana pengaruh parameter k (jumlah tetangga terdekat) terhadap hasil prediksi.

Dalam konteks ini, penulisan makalah ini bertujuan untuk memberikan pemahaman yang komprehensif tentang algoritma k-NN melalui pendekatan "from scratch." Dengan memahami langkah-langkah dasar yang diperlukan untuk mengimplementasikan algoritma ini secara manual, pembaca akan memiliki dasar yang kuat dalam menerapkan algoritma machine learning lainnya dan juga dapat melakukan penyesuaian sesuai dengan kebutuhan khusus dari setiap tugas.

Selain itu, dengan membangun algoritma k-NN dari awal, akan lebih mudah untuk mengidentifikasi potensi kelemahan dan kekuatan dari algoritma ini. Hal ini penting untuk dapat mengambil keputusan yang tepat dalam memilih algoritma yang paling sesuai untuk permasalahan tertentu. Dengan demikian, makalah ini akan memberikan wawasan yang mendalam tentang bagaimana k-NN beroperasi, bagaimana mengimplementasikannya secara praktis, serta bagaimana menerapkan dan memahami konsep dasar dalam machine learning secara lebih luas.

Melalui pemaparan ini, diharapkan pembaca akan mendapatkan pemahaman yang lebih mendalam tentang konsep inti algoritma k-NN dan mendapatkan gambaran yang lebih jelas tentang bagaimana algoritma ini dapat digunakan untuk mengolah dan menganalisis data dalam berbagai konteks.

## Keuntungan dan Kelemahan Model k-Nearest Neighbors (k-NN)

Algoritma k-Nearest Neighbors (k-NN) merupakan salah satu metode machine learning yang sederhana namun memiliki keunggulan dan juga keterbatasan. Pada bab ini, kami akan membahas secara rinci tentang keuntungan dan kelemahan dari model k-NN, memberikan gambaran yang lebih jelas tentang situasi di mana model ini paling efektif dan kapan perlu berhati-hati dalam penggunaannya.

### Keuntungan Model k-NN:

1. Sederhana dan Intuitif:
   
Salah satu keuntungan utama dari model k-NN adalah kesederhanaan konsepnya. Algoritma ini hanya memerlukan langkah-langkah dasar, yaitu menghitung jarak antara data baru dengan data pelatihan yang ada. Hal ini membuatnya mudah dipahami dan diimplementasikan oleh para pemula sekalipun.

2. Tidak Bergantung pada Asumsi Distribusi Data:
   
Model k-NN tidak membuat asumsi tertentu tentang distribusi data. Ini membuatnya sangat berguna dalam situasi di mana data memiliki karakteristik yang kompleks atau tidak mengikuti distribusi tertentu. K-NN cenderung lebih fleksibel dalam menangani data yang beragam.

3. Pembaruan Model Mudah:
   
Model k-NN dapat diperbarui dengan mudah saat ada data baru yang tersedia. Karena model ini tidak melibatkan proses pembelajaran berulang, Anda hanya perlu menambahkan data baru ke dalam dataset pelatihan yang ada.

4. Efektif untuk Data yang Tidak Terlalu Besar:
   
K-NN dapat memberikan hasil yang baik pada dataset yang relatif kecil. Ini membuatnya cocok untuk aplikasi di mana data terbatas, seperti pengenalan pola, analisis geospasial, dan lainnya.

### Kelemahan Model k-NN:

1. Sensitif terhadap Data Outlier:
   
Model k-NN rentan terhadap pengaruh data outlier, karena data outlier dapat mempengaruhi pemilihan tetangga terdekat. Ini dapat mengakibatkan hasil prediksi yang tidak akurat.

2. Kinerja yang Lambat pada Data Besar:
   
Proses pencarian tetangga terdekat pada dataset besar dapat memakan waktu yang lama. K-NN cenderung memiliki kinerja yang lambat ketika dihadapkan pada data dengan dimensi tinggi atau jumlah data yang besar.

3. Pentingnya Memilih Parameter k dengan Tepat:
   
Nilai k (jumlah tetangga terdekat) harus dipilih dengan hati-hati. Jika nilai k terlalu kecil, model bisa cenderung overfitting. Jika nilai k terlalu besar, model bisa cenderung underfitting.

4. Tidak Mampu Menangani Fitur Irrelevant:
   
K-NN tidak memiliki mekanisme bawaan untuk mengatasi fitur yang tidak relevan atau berlebihan. Ini dapat menyebabkan performansi model menurun jika fitur-fitur semacam itu ada dalam dataset.

5. Pengaruh Terhadap Data yang Tidak Normal:
    
K-NN rentan terhadap perubahan skala dan bentuk distribusi data. Jika data tidak terdistribusi secara normal, atau memiliki skala yang berbeda-beda, k-NN dapat memberikan hasil prediksi yang buruk.

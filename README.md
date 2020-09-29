# Adversarial Examples in Deep Learning for Multivariate Time Series Regression

# Introduction
Multivariate time series (MTS) regression tasks are common in many real-world data mining applications including finance, cybersecurity, energy, healthcare, prognostics, and many others. Due to the tremendous success of deep learning (DL) algorithms in various domains including image recognition and computer vision, researchers started adopting these techniques for solving MTS data mining problems, many of which are targeted for safety-critical and cost-critical applications. Unfortunately, DL algorithms are known for their susceptibility to adversarial examples which also makes the DL regression models for MTS forecasting also vulnerable to those attacks. To the best of our knowledge, no previous work has explored the vulnerability of DL MTS regression models to adversarial time series examples, which is an important step, specifically when the forecasting from such models is used in safety-critical and cost-critical applications. In this work, we leverage existing adversarial attack generation techniques from the image classification domain and craft adversarial multivariate time series examples for three state-of-the-art deep learning regression models, specifically Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). We evaluate our study using Google stock and household power consumption dataset. The obtained results show that all the evaluated DL regression models are vulnerable to adversarial attacks, transferable, and thus can lead to catastrophic consequences in safety-critical and cost-critical domains, such as energy and finance. 

# Dataset
Power consumption dataset: In this work, we evaluate the impact of adversarial attacks on household energy forecasting using the individual household electric power consumption dataset. The household power consumption dataset is a multivariate time series dataset that includes the measurements of electric power consumption in one household with a one-minute sampling rate for almost 4 years (December 2006 to November 2010) and collected via sub-meters placed in three distinct areas. The dataset is comprised of seven variables (besides the date and time) which includes global active power, global reactive power, voltage, global intensity, and sub-metering (1 to 3). We re-sample the dataset from minutes to hours and then predict global active power using seven variables or input features (global active power, global reactive power, voltage, global intensity, and sub-metering (1 to 3)). Then we use the first three years (2006 to 2009) for training our three DL models (LSTM, GRU, and CNN), and last year's data to test our models.

Google stock dataset: In this work, we evaluate the impact of adversarial attacks on Google stock prediction using the Google stock dataset. The Google stock dataset contains Google stock prices for the past 5 years. This multivariate time series dataset has six variables namely date, close, open, volume, high, and low. We use 30\% of the latest stock data as our test dataset and we train our three DL models (LSTM, GRU, and CNN) on the remaining 70\% of the data. To predict the Google stock prices, we consider the average stock prices and volume of the stocks traded from the previous days as input features. As the Google stock price prediction is dependant on multiple input features, it is a multivariate regression problem. We utilize the past 60 days of data to predict the stock price of the next day.

# DL algorithms
Power consumption dataset: The DL architecture of the DL models can be represented as LSTM(100,100,100) lh(14), GRU(100,100,100) lh(14), and CNN(60,60,60) lh(14). The notation LSTM(100,100,100) lh(14) refers to a network that has 100 nodes in the hidden layers of the first LSTM layer, 100 nodes in the hidden layers of the second LSTM layer, 100 nodes in the hidden layers of the third LSTM layer, and a sequence length of 14. In the end, there is a 1-dimensional output layer. In Fig. 1, we compare the performance of these three DL architectures in terms of their root mean squared error (RMSE). From Fig. 1, it is evident that the LSTM(100, 100, 100) has the best performance (with least RMSE) when predicting the global active power (without attack). The hyperparameter settings for the evaluated DL models are shown in Table I. 

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/Table1.PNG" height="150" width="600">
  </div>

Google stock dataset: The architectures of our DL models can be represented as LSTM(30,30,30) lh(60), GRU(30,30,30) lh(60), and CNN(60,60,60) lh(60). From Fig. 2, it is evident that the GRU(30, 30, 30) has the best performance (with least RMSE) when predicting stock opening prices (without attack) which was trained with 300 epochs using Adam optimizer and grid search for hyperparameter optimization to minimize the objective cost function: mean squared error (MSE). The hyperparameter settings for the evaluated DL models are shown in Table 1.

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/Comp.PNG" height="790" width="1257">
</div>

# Attack signatures: 
Power consumption dataset: Fig. 3 shows an example of the normalized FGSM and BIM attack signatures (adversarial examples) generated for the global reactive power variable (an input feature in the form of a time series). Similar adversarial examples are generated for the remaining five input features to evaluate their impact on the LSTM, GRU and CNN models for energy consumption prediction (global active power prediction). As shown in Fig. 3, the adversarial attack generated using BIM is close to the original time series data which makes such attack stealthy, hard to detect and often bypass the attack detection algorithms. 

Google stock dataset: Fig. 4 shows an example of the normalized  FGSM and BIM attack signatures (adversarial examples) generated for the volume of stocks traded (an input feature in the form of a time series). Similar adversarial examples are also generated for other input features to evaluate their impact on the LSTM, GRU and CNN models for the Google stock prediction (stock opening price). From Fig. 4, we observe that the adversarial attack generated using BIM is close to the original time series data, which makes such attacks hard to detect and thus have high chances of bypassing the attack detection methods.

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/Sig.PNG" height="810" width="1216">
  </div>

# Impact of adversarial attacks on MTSR system
Power consumption dataset: The impact of the generated adversarial examples on the household electric power consumption dataset is shown in Fig. 5. For the FGSM attack (with epsilon=0.2), we observe that the RMSE for the CNN, LSTM and GRU model (under attack) are increased by 19.9%, 12.3%, and 11%, respectively, when compared to the models without attack. For the BIM attack (with alpha=0.001, epsilon=0.2, and I=200), we also observe the similar trend, that is the RMSE of the CNN, LSTM and GRU models increased in a similar fashion, specifically by 25.9%, 22.9%, and 21.7%, respectively for the household electric power consumption dataset. We observe that for both FGSM and BIM attacks, it is evident that the CNN model is more sensitive to adversarial attacks when compared to the other DL models. Also, BIM results in a larger RMSE when compared to the FGSM. This means BIM is not only stealthier that FGSM, but also has a stronger impact on DL regression models for the this dataset.

Google stock dataset: The impact of the crafted adversarial examples on the Google stock dataset is shown in Fig. 6. For the FGSM attack (with epsilon=0.2), we observe that the RMSE for the CNN, LSTM and GRU model (under attack) are increased by 16%, 12.9%, and 13.1%, respectively, when compared to the models without attack. For the BIM attack (with alpha = 0.001, epsilon = 0.2 and I= 200), we also observe the similar trend, that is the RMSE for the CNN, LSTM and GRU model (under attack) are increased by 35.2%, 27.2% and 28.9%, respectively. Similar to our observation on the power consumption dataset, we notice that the CNN model is more sensitive to adversarial attacks when compared to the other DL models. Moreover, we also observe that BIM results in a larger RMSE when compared to the FGSM. 

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/Attack.PNG" height="820" width="1119">
  </div>

# Performance variation vs. the amount of perturbation
In Fig. 7, we evaluate the LSTM and GRU regression model's performance with respect to the different amount of perturbations allowed for crafting the adversarial MTS examples. We pick the LSTM and GRU as they showed the best performance for the MTS regression task in Fig. 1 and Fig. 2. We observe that for larger values of epsilon, FGSM is not very helpful in generating adversarial MTS examples for fooling the LSTM and GRU regression model. In comparison, with larger values of epsilon, BIM crafts more devastating adversarial MTS examples for fooling both the regression models and thus RMSE follows an increasing trend. This is due to the that BIM adds a small amount of perturbation alpha on each iteration whereas FGSM adds epsilon amount of noise for each data point in the MTS that may not be very helpful in generating inaccurate forecasting with higher RMSE values.

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/PerformanceVs.PNG" height="380" width="530">
  </div>

# Transferability of adversarial examples
To evaluate the transferability of adversarial attacks, we apply the adversarial examples crafted for a DL MTS regression model on the other DL models. Table II summarizes the obtained results on transferability. We observe that for both datasets, the adversarial examples crafted for CNN are the most transferable. This means a higher RMSE is observed when adversarial examples crafted for the CNN model are transferred to other models. For instance, adversarial MTS examples crafted using BIM for the CNN regression model (Google stock dataset) causes a 23.4% increase when transferred to the GRU regression model. A similar trend is also observed, however, with a lower percentage increases, when adversarial examples crafted for GRU and LSTM regression models are transferred to the other DL regression models. In addition, the obtained results also show that BIM is better than FGSM in fooling (even when they are transferred) the DL models for MTS regression tasks, e.g. BIM increases the RMSE more when compared to the FGSM. Overall, the results show that the adversarial examples are capable of generalizing to a different DL network architecture. This type of attack is known as black box attacks, where the attackers do not have access to the target model’s internal parameters, yet they are able to generate perturbed time series that fool the DL models for MTSR tasks.

<div align="center">
<img src="https://github.com/dependable-cps/adversarial-MTSR/blob/master/Images/Table2.PNG" height="200" width="1139">
  </div>

# Citation
If this is useful for your work, please cite our <a href="https://arxiv.org/abs/2009.11911">paper</a>:<br>

<div class="highlight highlight-text-bibtex"><pre>
@article{advMTSR,
title={Adversarial Examples in Deep Learning for Multivariate Time Series Regression},
  author={Mode, Gautam Raj and Hoque, Khaza Anuarul},
  journal={arXiv preprint arXiv:2009.11911},
  year={2020}
}
</pre></div>

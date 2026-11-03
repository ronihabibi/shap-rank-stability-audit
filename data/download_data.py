"""Downloads the five public datasets into data/raw/ (needs internet). Run: python data/download_data.py"""
import os,urllib.request,hashlib
HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,'raw'); os.makedirs(RAW,exist_ok=True)
SRC={
 'adult-all.csv':'https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv',
 'german.csv':'https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv',
 'Telco-Customer-Churn.csv':'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
 'diamonds.csv':'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv',
 'insurance.csv':'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'}
for f,u in SRC.items():
    p=os.path.join(RAW,f)
    if not os.path.exists(p): urllib.request.urlretrieve(u,p)
    print(f, os.path.getsize(p),'bytes  sha256',hashlib.sha256(open(p,'rb').read()).hexdigest()[:16])

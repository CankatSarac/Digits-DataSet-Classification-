#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Standat Python importlarimizi yaptik
import matplotlib.pyplot as plt

# Datasetimi importladim
from sklearn import datasets, svm, metrics

# digits degiskenine datasetimi atadim
digits = datasets.load_digits()

# Ilgilendigimiz data 8x8lik formatta 
# Once ilk 4 resme bakalim
# Eger imagefile uzerinden calisacaksak oncelikle matplotlib.pyplot.imread. komutunu cagirmaliyiz
# Butun resimlerin ayni formatta olduguna dikkat ediyorum
# Hangi resmin hangi sayiyi temsil ettigini bildigim degiskenlere target ismini atadim
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# Classify uygulamak icin resimleri yassilastiriyorum (flatten image) tek coloumn yapiyorumda denilebilir
# Datayi matrixe cevirdim.
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Classifiera atadim
classifier = svm.SVC(gamma=0.001)

# Digitleri train sete atadim 
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Ikinci asamada test datasi tahmin etti
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


# In[ ]:





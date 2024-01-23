import cv2
import torch
import requests

# Kamera akışını başlat
cap = cv2.VideoCapture(0)

# ResNet18 modelini yükle (ImageNet üzerinde önceden eğitilmiş)
model = torch.hub.load('pytorch/vision:v0.11.1', 'resnet18', pretrained=True)
model.eval()

# Sınıf etiketlerini yükle (ImageNet sınıf etiketleri)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = response.json()

# ResNet18 için görüntü ön işleme fonksiyonunu tanımla
preprocess = torch.hub.load('pytorch/vision:v0.11.1', 'imagenet', 'resnet18', pretrained=True).transform

while True:
    # Kamera görüntüsünü yakala
    ret, frame = cap.read()

    # Görüntüyü ResNet18 için uygun formata dönüştür
    img = preprocess(frame)
    img = img.unsqueeze(0)

    # Tahmin yap
    with torch.no_grad():
        predictions = model(img)

    # En yüksek tahmin sonucunu al
    _, predicted_class = torch.max(predictions.data, 1)
    confidence = torch.nn.functional.softmax(predictions, dim=1)[0][predicted_class.item()].item()

    # Etiketi al
    label = labels[predicted_class.item()]

    # Çerçeve üzerine meyve bilgisini yaz
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Kameradan alınan görüntüyü göster
    cv2.imshow('Meyve Tanıma', frame)

    # Çıkış için "q" tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışını serbest bırak ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()

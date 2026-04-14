import tensorflow as tf
import numpy as np
import cv2

# LOAD MODEL CNN
model = tf.keras.models.load_model(
    "model_cnn_penyakit_daun_testset.h5"
)

# INFORMASI PENYAKIT
class_info = {
    0: {
        "nama": "Bercak Hitam pada Daun",
        "penjelasan": "Daun muncul bercak hitam kecil akibat bakteri.",
        "saran": "Kurangi air berlebih dan semprot obat antibakteri."
    },
    1: {
        "nama": "Busuk Daun Awal",
        "penjelasan": "Daun muncul bercak coklat dan mulai mengering.",
        "saran": "Buang daun sakit dan semprot obat jamur."
    },
    2: {
        "nama": "Busuk Daun Parah",
        "penjelasan": "Daun cepat layu dan menghitam, menyebar cepat.",
        "saran": "Segera semprot fungisida dan pisahkan tanaman sakit."
    },
    3: {
        "nama": "Jamur pada Daun",
        "penjelasan": "Daun menguning dan bagian bawah berjamur.",
        "saran": "Kurangi kelembaban dan semprot obat jamur."
    },
    4: {
        "nama": "Bercak Abu-Abu pada Daun",
        "penjelasan": "Bercak kecil abu-abu menyebar di daun.",
        "saran": "Buang daun sakit dan semprot fungisida."
    },
    5: {
        "nama": "Serangan Kutu Daun",
        "penjelasan": "Daun menguning akibat gigitan kutu kecil.",
        "saran": "Gunakan obat pembasmi hama."
    },
    6: {
        "nama": "Bercak Lingkaran pada Daun",
        "penjelasan": "Bercak bulat seperti sasaran tembak.",
        "saran": "Semprot fungisida dan jaga sirkulasi udara."
    },
    7: {
        "nama": "Daun Kuning dan Keriting",
        "penjelasan": "Daun menguning, keriting, tanaman kerdil.",
        "saran": "Cabut tanaman sakit dan kendalikan kutu putih."
    },
    8: {
        "nama": "Daun Belang-Belang",
        "penjelasan": "Daun belang hijau muda dan tua.",
        "saran": "Cabut tanaman sakit dan bersihkan alat tanam."
    },
    9: {
        "nama": "Daun Sehat",
        "penjelasan": "Daun hijau segar tanpa bercak.",
        "saran": "Lanjutkan perawatan rutin."
    }
}

# PARAMETER SISTEM
CONFIDENCE_THRESHOLD = 60      # persen
MIN_GREEN_RATIO = 0.05         # minimal area hijau 5%


# FUNGSI DETEKSI ADA DAUN
def ada_daun(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna hijau daun
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / mask.size

    return green_ratio > MIN_GREEN_RATIO

# ==========================
# BUKA KAMERA
# ==========================
cap = cv2.VideoCapture(0)

print("📷 Kamera aktif")
print("Arahkan daun tomat langsung ke kamera")
print("Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # CROP TENGAH (KURANGI NOISE)
    h, w, _ = frame.shape
    crop_size = min(h, w)
    start_x = w // 2 - crop_size // 2
    start_y = h // 2 - crop_size // 2
    crop = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # CEK ADA DAUN ATAU TIDAK
    if not ada_daun(crop):
        nama = "Tidak Ada Daun"
        penjelasan = "Arahkan kamera ke daun tomat."
        saran = "Pastikan daun terlihat jelas dan memenuhi layar."
        confidence = 0
        color = (0, 0, 255)

    else:
        # ==========================
        # PREPROCESS
        # ==========================
        img = cv2.resize(crop, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # ==========================
        # PREDIKSI CNN
        # ==========================
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred) * 95

        if confidence < CONFIDENCE_THRESHOLD:
            nama = "Daun Tidak Jelas"
            penjelasan = "Daun terlalu jauh atau kurang fokus."
            saran = "Dekatkan kamera dan pastikan pencahayaan cukup."
            color = (0, 0, 255)
        else:
            info = class_info[class_id]
            nama = info["nama"]
            penjelasan = info["penjelasan"]
            saran = info["saran"]
            color = (0, 255, 0)

    # ==========================
    # TAMPILKAN HASIL
    # ==========================
    cv2.putText(frame, f"Penyakit : {nama}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Keyakinan : {confidence:.1f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"Keterangan : {penjelasan}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Saran : {saran}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Deteksi Penyakit Daun Tomat", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

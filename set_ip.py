from pypylon import pylon
import time

def force_camera_ip():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    if not devices:
        print("ยังไม่พบกล้องในระบบ ลองเช็คไฟกล้องอีกครั้งครับ")
        return

    for device in devices:
        print(f"พบกล้อง: {device.GetFriendlyName()}")
        # สั่ง Force IP ไปที่ 192.168.10.2 (เพราะ Pi เป็น .1)
        # ใช้ Mac Address ของกล้องที่ตรวจเจอ
        success = tl_factory.CreateTl('BaslerGigE').ForceIp(device.GetEthernetAddress(), "192.168.10.2", "255.255.255.0", "0.0.0.0")
        if success:
            print("Force IP สำเร็จ! ตอนนี้กล้องควรจะเป็น 192.168.10.2")
        else:
            print("Force IP ไม่สำเร็จ")

if __name__ == "__main__":
    force_camera_ip()
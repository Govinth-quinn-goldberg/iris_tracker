
import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for BLE devices for 15 seconds...\n")

    devices = await BleakScanner.discover(timeout=15)

    for device in devices:
        print(f"Name: {device.name!r} | Address: {device.address}")

    print("\nScan complete.")

asyncio.run(main())

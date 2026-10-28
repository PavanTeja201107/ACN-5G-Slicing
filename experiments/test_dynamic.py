from controller.controller import dynamic_reallocation


def main():

    # Initial static allocation
    allocation = {
        "embb": 50,
        "urllc": 30,
        "iot": 20,
    }

    # Simulated traffic demand
    demand = {
        "embb": 20,
        "urllc": 45,
        "iot": 5,
    }

    print("Dynamic resource allocation test")
    print("=" * 45)

    print()
    print("Initial allocation:")
    print(allocation)

    print()
    print("Traffic demand:")
    print(demand)

    print()
    print("Running controller...")

    new_allocation, changed, _, _ = dynamic_reallocation(
        demand,
        allocation,
        step=5,
    )

    print()
    print("New allocation:")
    print(new_allocation)

    print()
    print("Reallocation performed:")
    print(changed)


if __name__ == "__main__":
    main()
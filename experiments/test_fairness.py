from controller.fairness import allocation_fairness


def main():

    allocations = {
        "static_50_30_20": {
            "embb": 50,
            "urllc": 30,
            "iot": 20,
        },

        "balanced": {
            "embb": 33,
            "urllc": 33,
            "iot": 34,
        },

        "urlcc_priority": {
            "embb": 20,
            "urllc": 60,
            "iot": 20,
        },

        "highly_unbalanced": {
            "embb": 90,
            "urllc": 9,
            "iot": 1,
        },
    }

    print("Fairness evaluation")
    print("=" * 40)

    for name, allocation in allocations.items():

        fairness = allocation_fairness(allocation)

        print()
        print(name)
        print("Allocation:", allocation)
        print(f"Fairness index: {fairness:.4f}")


if __name__ == "__main__":
    main()
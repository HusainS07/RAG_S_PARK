import json
from datetime import datetime


GROUND_TRUTH_BY_NUMBER = {
    1: ("Open Profile > My Bookings, choose the booking, open its detail view, tap Cancel Booking, and confirm the dialog. After confirmation the slot is released and refund handling starts according to the cancellation policy.", ["3.4", "3.6", "8.2"]),
    2: ("Go to Profile > My Bookings, tap the reservation, select Cancel Booking, and confirm. The system sends cancellation confirmation by email and processes any refund according to policy.", ["8.2"]),
    3: ("The document says a booking after its start time receives 0 percent refund and is non-refundable once active. It still lists Active bookings as able to move to Cancelled, but cancellation after start does not produce a refund.", ["7.2", "8.1"]),
    4: ("Use Profile > My Bookings to select the booking and tap Cancel Booking. The guide does not describe deleting a booking record from history; cancelled bookings remain records with cancelled status.", ["3.4", "8.2"]),
    5: ("The Cancel Booking button is in the booking detail view, reached from Profile > My Bookings by tapping the booking.", ["3.4", "8.2"]),
    6: ("Refund amount depends on how far before the booking start time cancellation happens: more than 24 hours gets 100 percent, 6-24 hours gets 75 percent, 1-6 hours gets 50 percent, less than 1 hour gets 0 percent, and after start gets 0 percent.", ["8.1"]),
    7: ("Once cancellation is confirmed, the slot is immediately released back to the available pool in MongoDB and Redis, a cancellation confirmation is sent by email, and refund processing begins.", ["3.6", "8.2"]),
    8: ("Active bookings are listed in the Bookings tab on the Profile page.", ["3.4", "10.1"]),
    9: ("Booking history is accessed from the Profile page's Bookings tab, which lists current and past bookings.", ["3.4", "10.1"]),
    10: ("Past bookings are stored in the user's booking history, accessible from Profile > Bookings.", ["3.4", "7.1"]),
    11: ("Use the Profile page Bookings tab to view all current and past parking reservations.", ["3.4", "10.1"]),
    12: ("The Bookings tab shows the user's current bookings, so upcoming or active reservations are found there.", ["3.4"]),
    13: ("The document names the section as the Bookings tab on the Profile page, also described as Profile > My Bookings for cancellation.", ["3.4", "8.2", "10.1"]),
    14: ("An active booking entry shows lot name, slot number, booking date and time, duration, amount paid, and current status.", ["3.4"]),
    15: ("A booking confirmation includes a unique Booking ID. Confirmation email content includes Booking ID, lot name, slot, date/time, and amount paid.", ["3.3", "7.1", "12.1"]),
    16: ("Yes. Lot cards on the Book page show the lot name, address, price per hour, and currently available slots before booking.", ["3.2"]),
    17: ("The document does not list amenity details as a supported pre-booking field. It explicitly says lot cards show name, physical address, price per hour, and available slot count.", ["3.2"]),
    18: ("The physical location address is shown on each lot card on the Book page before booking.", ["3.2"]),
    19: ("The guide does not describe separate parking instructions in a booking. It specifies confirmation details such as Booking ID, lot name, slot, date/time, and amount paid.", ["12.1"]),
    20: ("The document says lot-operator cancellation terms may be displayed on the lot detail page before confirmation. It does not describe a separate rules and restrictions page.", ["8.1"]),
    21: ("Yes. After cancellation, a cancellation confirmation is sent to the user's email.", ["8.2", "12.1"]),
    22: ("Yes. The document says a cancellation confirmation email is sent when cancellation is complete.", ["8.2", "12.1"]),
    23: ("Cancellation is notified by email; the cancellation email includes confirmation and refund amount or timeline.", ["8.2", "12.1"]),
    24: ("Yes. The document explicitly says a cancellation confirmation is sent to the registered email address.", ["8.2", "12.1"]),
    25: ("The guide does not describe a separate cancellation-status tracker. It says cancellation confirmation is emailed and refund processing follows the policy.", ["8.2", "12.1"]),
    26: ("Refund handling depends on payment method. Wallet-paid bookings are refunded to wallet, usually within minutes or instantly. Razorpay-paid bookings are refunded to the original payment method and can take from instant to 7 business days depending on method.", ["5.5", "8.4"]),
    27: ("Cancellation refunds are tiered by time before booking start: more than 24 hours 100 percent, 6-24 hours 75 percent, 1-6 hours 50 percent, less than 1 hour 0 percent, after start 0 percent.", ["8.1"]),
    28: ("A full refund applies only when cancellation is more than 24 hours before the booking start time, unless a lot operator has different displayed terms.", ["8.1"]),
    29: ("The policy retains part or all of the amount for later cancellations: 6-24 hours keeps an administrative fee, 1-6 hours refunds half, and less than 1 hour or after start gives no refund.", ["8.1"]),
    30: ("The refund is 100 percent more than 24 hours before start, 75 percent 6-24 hours before, 50 percent 1-6 hours before, 0 percent less than 1 hour before, and 0 percent after start.", ["8.1"]),
    31: ("Yes, late cancellation can reduce or remove the refund. The document describes 75 percent, 50 percent, and 0 percent refund tiers depending on timing.", ["8.1"]),
    32: ("If cancelled less than 1 hour before start, the refund is 0 percent. After booking start, the booking is also non-refundable.", ["8.1"]),
    33: ("Yes. Partial refunds are defined: 75 percent for cancellations 6-24 hours before start and 50 percent for cancellations 1-6 hours before start.", ["8.1"]),
    34: ("Use the Help page or Contact form. Include details such as Booking ID, lot name, date of issue, and screenshots so support can resolve the cancellation issue.", ["11.1", "11.2", "11.3"]),
    35: ("Contact support through the Help page Contact form, which collects name, email, subject, and message and posts to /api/contact.", ["11.2"]),
    36: ("If Cancel Booking does not work, submit a support inquiry from the Help page Contact form with Booking ID, lot name, issue date, and screenshots.", ["11.2", "11.3"]),
    37: ("The support team handles cancellation problems submitted through the Help page Contact form.", ["11.2", "11.3"]),
    38: ("The document does not provide a customer support phone number. It directs users to the in-app Help page or Contact form.", ["11.1", "11.2"]),
    39: ("For booking cancellation issues, expected first response is 4 business hours and target resolution is 1 business day.", ["11.3"]),
    40: ("The document mentions a Contact form and Help page, not live chat. Users should submit the cancellation issue through the Contact form.", ["11.2"]),
    41: ("Open the Book page, browse approved lots, select a lot, choose an available slot, select duration and payment method, then confirm payment. The booking is confirmed after wallet deduction or Razorpay webhook verification.", ["3.2", "3.3", "7.1"]),
    42: ("The booking flow is: browse lots, select a lot, view the slot grid, select an available slot, tap Book, provide duration and payment method, complete payment, then receive a Booking ID and email confirmation.", ["7.1"]),
    43: ("Search for parking on the Book page, which displays approved parking lots and lets users narrow lots by location or price range.", ["3.2", "7.1"]),
    44: ("Yes. The platform is designed for users to pre-book parking slots. The document does not state a maximum advance booking window.", ["1.3", "3.2", "7.1"]),
    45: ("The document says users can pre-book a slot but does not specify how far in advance reservations can be made.", ["1.3", "7.1"]),
    46: ("The booking API uses slot ID, user email, desired duration, and chosen payment method. Users should also ensure the vehicle registration number matches the vehicle that will park.", ["7.1", "15"]),
    47: ("The document does not give a total booking-process duration. It says slot availability is real time and payment methods are generally processed in seconds, while booking is confirmed after payment verification.", ["6.5", "7.1", "9.1"]),
    48: ("Yes. The Book page includes filter controls to narrow lots by location or price range.", ["3.2"]),
    49: ("The documented filter options are location and price range. Other filters are not specified in the guide.", ["3.2"]),
    50: ("Yes. Users can narrow lots by price range on the Book page.", ["3.2"]),
    51: ("The document says users can filter by location, but it does not explicitly describe a distance-radius filter.", ["3.2"]),
    52: ("The guide does not state that amenity filters are available. It only explicitly documents location and price-range filters.", ["3.2"]),
    53: ("The guide does not mention an EV charging search or filter.", ["3.2"]),
    54: ("The guide does not mention an accessibility filter.", ["3.2"]),
    55: ("The guide does not mention filtering by parking type. It documents location and price-range filters only.", ["3.2"]),
    56: ("Before confirming, check the lot name, physical address, price per hour, available slot count, selected slot, duration, payment method, vehicle registration, and any lot-operator terms shown before confirmation.", ["3.2", "7.1", "8.1", "10.3"]),
    57: ("Verify parking details from the lot card and slot grid: address, rate, slot availability/status, selected slot, booking date/time, duration, and vehicle registration.", ["3.2", "3.3", "10.3"]),
    58: ("A grounded checklist is: confirm lot/address, check price per hour, select an available green slot, verify duration and payment method, ensure vehicle plate is correct, and review displayed lot terms.", ["3.2", "3.3", "8.1", "10.3"]),
    59: ("Yes. Correct vehicle registration is important because lot operators use the license plate to verify the booked vehicle and ANPR cameras at some facilities can flag mismatches.", ["10.3", "15"]),
    60: ("The guide says lot cards show address, price, and available slot count, and lot-operator terms may override cancellation defaults and are displayed before confirmation.", ["3.2", "8.1"]),
    61: ("After successful booking, the user receives an immediate confirmation screen with a unique Booking ID and a confirmation email containing Booking ID, lot name, slot, date/time, and amount paid.", ["3.3", "7.1", "12.1"]),
    62: ("Yes. A unique Booking ID is generated and shown immediately after payment confirmation.", ["3.3", "7.1", "15"]),
    63: ("The document does not mention a QR code for booking confirmation. It specifies a Booking ID and email confirmation.", ["3.3", "12.1"]),
    64: ("The documented confirmation format is an on-screen Booking ID plus an email containing Booking ID, lot name, slot, date/time, and amount paid.", ["3.3", "12.1"]),
    65: ("The document does not say that booking confirmation can be downloaded. It says confirmation appears on screen and is emailed.", ["3.3", "12.1"]),
    66: ("Access booking confirmation details later from the Profile page Bookings tab, where current and past booking records are listed.", ["3.4", "10.1"]),
    67: ("Confirmation is important because it proves the booking was confirmed and provides the Booking ID, lot, slot, date/time, and amount paid. If confirmation does not arrive, the user should check spam or contact support with transaction reference.", ["12.1", "15"]),
    68: ("The document does not explicitly state that confirmation is required for entry. It identifies Booking ID and confirmation email as proof of a confirmed booking and support reference.", ["12.1", "15"]),
    69: ("If confirmation does not arrive or is lost, check the Profile Bookings tab and email spam folder; if needed, contact support with the transaction reference from the payment app.", ["3.4", "15"]),
    70: ("The guide allows booking for someone else if the vehicle registration number entered during booking matches the vehicle that will actually park. It does not separately describe sharing confirmation.", ["15"]),
    71: ("Yes, if the lot operator allows it. From the booking detail page, tap Extend Booking, select additional time, and pay the extra amount.", ["3.5", "8.3"]),
    72: ("Add time from the booking detail page by selecting Extend Booking, choosing additional time, and paying the extra amount.", ["8.3"]),
    73: ("Yes. Duration extensions are available from the booking detail page in the app if the lot operator allows them.", ["8.3"]),
    74: ("Open booking details, choose Extend Booking, select extra duration, pay the additional amount, and after payment confirmation the slot is locked for the extended period.", ["8.3"]),
    75: ("The document does not specify a maximum number of extensions.", ["8.3"]),
    76: ("The document does not specify a general extension-duration limit. Extensions depend on whether the lot operator allows them and on paying for additional time.", ["3.5", "8.3"]),
    77: ("Extension cost is the extra amount for the additional selected time. The document does not give a fixed rate beyond lot pricing per hour.", ["3.2", "8.3"]),
    78: ("If the extension option is unavailable, the lot likely does not allow in-session modification; the system hides Modify for lots where the operator has disabled it.", ["3.5"]),
    79: ("You may be unable to extend because not all lots allow in-session modifications and the operator may have disabled the option.", ["3.5", "8.3"]),
    80: ("The guide does not specify formal alternatives when extension is not allowed. To avoid overstay, users should extend before end time if possible; otherwise they may need support or a separate available booking.", ["15"]),
    81: ("The document does not explicitly describe using a new booking instead of extension, but booking another available slot follows the normal booking flow if a slot is available.", ["7.1"]),
    82: ("Extensions may not be allowed because the parking facility or lot operator has disabled in-session modifications.", ["3.5"]),
    83: ("Extension is not permitted when the lot operator does not allow in-session modifications or when the option is hidden by the system.", ["3.5"]),
    84: ("Yes. The document says not all lots allow in-session modifications, so there are cases where extension is unavailable.", ["3.5"]),
    85: ("The document does not describe a Find My Car feature. It provides lot address, slot number, and booking details that can help locate the vehicle manually.", ["3.4"]),
    86: ("No Find My Car feature is documented in the guide.", ["3.4"]),
    87: ("The guide does not specify a vehicle location feature or its accuracy.", ["3.4"]),
    88: ("The guide does not describe technology for locating a parked vehicle. It only mentions ANPR for vehicle verification at some facilities.", ["10.3", "15"]),
    89: ("The document does not describe locating a car outside the facility through the app.", ["3.4"]),
    90: ("The guide does not give a Find My Car workflow. A user can rely on documented booking details such as lot address and slot number.", ["3.4"]),
    91: ("The document does not mention marking a parking location in the app.", ["3.4"]),
    92: ("The document does not mention adding notes to a parking location.", ["3.4"]),
    93: ("The document does not mention taking photos of a parking spot in the app.", ["3.4"]),
    94: ("The document does not specifically mention level or section notes, but booking entries include slot number and lot details.", ["3.4"]),
    95: ("If no Find My Car feature exists, use the documented booking details: lot address, slot number, booking date/time, and any facility information visible in the booking record.", ["3.4"]),
    96: ("Manual tracking options are not formally documented. Based on available records, the user can check lot address and slot number in the booking details.", ["3.4"]),
    97: ("If payment fails, try a different payment method, contact the bank if needed, or use the Smart Parking Wallet. Payment failure or overcharge support receives first response within 4 business hours.", ["15", "11.3"]),
    98: ("Common payment failure reasons include insufficient funds, card or UPI limits, expired card, or a bank security block on new merchants.", ["15"]),
    99: ("Retry by using another payment method or the Smart Parking Wallet. For Razorpay, booking is confirmed only after successful payment verification.", ["6.3", "7.1", "15"]),
    100: ("A booking is only confirmed after wallet deduction or Razorpay webhook verification. If payment is not completed, the pending order can be cancelled and temporary slot locks expire.", ["6.3", "6.4", "7.1", "9.2"]),
    101: ("Accepted payment methods are Smart Parking Wallet and Razorpay methods including UPI, credit/debit cards, net banking, RuPay card, and EMI.", ["6.5"]),
    102: ("The guide does not describe storing a backup payment method. It says users can choose wallet or Razorpay and try a different payment method if one fails.", ["3.3", "15"]),
    103: ("Report a double payment through the Help page Contact form with Booking ID, lot name, issue date, transaction reference, and screenshots. Payment overcharge issues have a 4 business hour first response and 1-2 business day resolution target.", ["11.2", "11.3"]),
    104: ("If charged twice, contact support through the Help page with transaction details and screenshots. This falls under payment failure or overcharge support.", ["11.2", "11.3"]),
    105: ("Use the Contact form and include name, email, subject, message, Booking ID, lot name, date of issue, transaction reference, and screenshots.", ["11.2"]),
    106: ("Payment failure or overcharge issues have an expected first response of 4 business hours and target resolution of 1-2 business days.", ["11.3"]),
    107: ("Wallet refunds are instant or within minutes. Razorpay UPI refunds take instant to 2 hours, card refunds take 3-5 business days, and net banking refunds take 3-7 business days.", ["5.5", "8.4"]),
    108: ("Refund timeline depends on payment method: wallet instant, UPI instant to 2 hours, card 3-5 business days, net banking 3-7 business days.", ["8.4"]),
    109: ("Wallet-paid bookings refund to wallet. Razorpay-paid bookings that qualify are refunded to the original payment method.", ["5.5", "8.4"]),
    110: ("The guide does not describe a dedicated refund-status tracker. Cancellation email includes refund amount or timeline, and support can be contacted if needed.", ["12.1"]),
    111: ("To avoid payment issues, ensure sufficient funds, card/UPI limits are adequate, card is not expired, bank is not blocking the merchant, and consider using the Smart Parking Wallet.", ["15"]),
    112: ("Before booking, check wallet or bank balance, choose a supported payment method, verify vehicle details, and review booking amount and lot details before confirming.", ["3.2", "6.5", "10.3", "15"]),
    113: ("Yes. Insufficient funds are listed as a common reason for payment decline, so users should check account or wallet balance before booking.", ["15"]),
    114: ("The platform verifies Razorpay payments server-side through webhook signature verification before confirming the booking. Users should ensure payment completes successfully in Razorpay checkout.", ["6.3", "7.1"]),
    115: ("First verify booking details: slot number, license plate, date, and time. If the booking details are correct, submit a dispute through the Help page with Booking ID and photographic evidence.", ["3.7"]),
    116: ("Verify a violation by checking the correct slot number, license plate, date, and time against your active booking details.", ["3.7"]),
    117: ("The guide does not list fields on a violation notice. It says users should verify slot number, license plate, date, and time.", ["3.7"]),
    118: ("Yes. The guide allows disputes if the user's booking details are correct, implying notices can be disputed when incorrect.", ["3.7"]),
    119: ("Check whether the notice matches the booked slot number, license plate, date, and time. If those are correct, dispute through the Help page.", ["3.7"]),
    120: ("Provide Booking ID and photographic evidence when disputing a parking violation.", ["3.7"]),
    121: ("Submit a dispute through the Help page and attach Booking ID plus photographic evidence. The support team liaises with the lot operator.", ["3.7"]),
    122: ("Provide Booking ID, photographic evidence, and relevant booking details such as slot number, license plate, date, and time.", ["3.7"]),
    123: ("The document says users can attach photographic evidence to a Help page dispute. It does not describe all upload formats.", ["3.7"]),
    124: ("The support team helps resolve disputes by liaising with the lot operator when the booking details are correct.", ["3.7"]),
    125: ("If the violation appears to be a system or operator mistake and booking details are correct, submit a dispute through Help; support works with the lot operator.", ["3.7"]),
    126: ("Yes. The app's Help page is used to submit parking violation disputes with Booking ID and evidence.", ["3.7", "11.1"]),
    127: ("If the violation was the user's fault, such as parking in the wrong slot, the user must resolve the fine directly with the relevant authority.", ["3.7"]),
    128: ("Parking incorrectly, such as using the wrong slot, is treated as the user's fault; the user must resolve any fine with the relevant authority.", ["3.7"]),
    129: ("The Wallet page shows current balance and full transaction history, including transaction type, amount, date, and booking or top-up reference.", ["5.2"]),
    130: ("The guide describes wallet transaction history on the Wallet page. It does not specifically mention payment statements.", ["5.2"]),
    131: ("The guide does not say users can download transaction records.", ["5.2"]),
    132: ("The guide does not state that transaction history is searchable.", ["5.2"]),
    133: ("The guide does not specify how far back transaction history can be viewed.", ["5.2"]),
    134: ("Yes. A confirmation dialog appears before cancellation to prevent accidental cancellations.", ["3.6", "8.2"]),
    135: ("Yes. The confirmation dialog functions as a warning before cancellation.", ["3.6"]),
    136: ("The guide does not describe undoing a cancellation. It says once confirmed, the slot is immediately released back to the available pool.", ["3.6"]),
    137: ("The Contact form collects name, email, subject, and message. Users should include Booking ID, lot name, issue date, and screenshots when relevant.", ["11.2"]),
    138: ("Prepare for support by collecting Booking ID, lot name, issue date, transaction reference if payment-related, screenshots, and a clear message.", ["11.2", "15"]),
    139: ("Provide as much detail as possible: Booking ID, lot name, date of issue, screenshots, and for payment issues a transaction reference.", ["11.2", "15"]),
    140: ("For booking-related support, the guide recommends including the Booking ID because vague requests take longer to resolve.", ["11.2"]),
    141: ("The document does not describe any QR code role in Smart Parking. It uses Booking ID and email confirmation as the documented confirmation artifacts.", ["3.3", "12.1"]),
    142: ("The guide does not explain QR code usage for parking. It documents Booking ID and confirmation email instead.", ["3.3", "12.1"]),
    143: ("The guide does not say a QR code is present in the booking. Confirmation includes Booking ID and email details.", ["3.3", "12.1"]),
    144: ("The guide does not mention QR codes or screenshotting them.", ["3.3", "12.1"]),
    145: ("Booking duration can be modified from the booking detail screen where the facility permits it. Extensions require selecting additional time and paying extra; reductions may follow refund policy.", ["3.5", "8.3"]),
    146: ("The documented change is booking duration. Extensions add paid time; reductions are treated as partial cancellations. The guide does not document changing all booking fields.", ["3.5", "8.3"]),
    147: ("The guide does not explicitly say the booking date can be changed. It describes modifying duration and extending the end time if allowed.", ["3.5", "8.3"]),
    148: ("The guide does not describe changing the parking location after booking. Users choose a lot and slot during the booking flow.", ["7.1", "8.3"]),
    149: ("Vehicle registration can be managed in the user's profile. The guide does not say a confirmed booking's vehicle info can be changed; correct plate registration is important before booking.", ["10.3"]),
    150: ("Open the Book page to view approved lots, choose a location/lot, then open its slot grid. Slot statuses are fetched in real time and shown as green for available, red for booked, and amber for maintenance.", ["3.2", "3.3", "7.1", "9.1"]),
}


def phase_for_index(index):
    if index < 50:
        return "simple"
    if index < 100:
        return "medium"
    return "complex"


def build_ground_truths():
    with open("expanded_eval_data.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    phases = {"simple": [], "medium": [], "complex": []}
    for index, item in enumerate(questions[:150]):
        number = index + 1
        answer, source_sections = GROUND_TRUTH_BY_NUMBER[number]
        phase = phase_for_index(index)
        phases[phase].append({
            "question_number": number,
            "phase": phase,
            "category": item["category"],
            "question": item["question"],
            "ground_truth": answer,
            "source": {
                "document": "Company_Data.pdf",
                "sections": source_sections,
            },
        })

    return {
        "metadata": {
            "source_document": "Company_Data.pdf",
            "question_source": "expanded_eval_data.json",
            "question_count": 150,
            "phase_ranges": {
                "simple": "1-50",
                "medium": "51-100",
                "complex": "101-150",
            },
            "created_at": datetime.now().isoformat(),
            "notes": (
                "Ground truths are derived from Company_Data.pdf. When the document does "
                "not specify a requested feature, the ground truth explicitly says so."
            ),
        },
        "phases": phases,
    }


if __name__ == "__main__":
    payload = build_ground_truths()
    with open("phase_ground_truths.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
    print("Wrote phase_ground_truths.json with 150 ground-truth answers")

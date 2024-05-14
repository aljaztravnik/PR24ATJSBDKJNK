import matplotlib.pyplot as plt
from csv import DictReader
import numpy as np
from datetime import datetime
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

warnings.filterwarnings("ignore")


class Data:
    def __init__(self):
        self.podatki = []
        dtype = [
            ('Date', 'U10'),  # Date as string
            ('Time', 'U10'),  # Time as string
            ('Location', 'U100'),  # Location as string
            ('Operator', 'U100'),  # Operator as string
            ('Flight', 'U100'),  # Flight number as string
            ('Route', 'U100'),  # Route as string
            ('Type', 'U100'),  # Aircraft type as string
            ('Registration', 'U100'),  # Registration as string
            ('cn_In', 'U100'),  # Construction number or other identifier as string
            ('Aboard', np.int32),  # Number of people aboard as integer
            ('Fatalities', np.int32),  # Number of fatalities as integer
            ('Ground', np.int32),  # Number of ground fatalities as integer
            ('Summary', 'U1000')  # Summary as string
        ]

        with open("podatki\\Airplane_Crashes_and_Fatalities_Since_1908.csv", 'r', encoding='utf-8') as file:
            csv_reader = DictReader(file)
            for row in csv_reader:
                date = row['Date']
                time = row['Time']
                location = row['Location']
                operator = row['Operator']
                flightNr = row['Flight #']
                route = row['Route']
                type = row['Type']
                registration = row['Registration']
                CnIn = row['cn/In']
                aboard = int(row["Aboard"]) if row["Aboard"] else 0
                fatalities = int(row["Fatalities"]) if row["Fatalities"] else 0
                ground = int(row['Ground']) if row["Ground"] else 0
                summary = row['Summary']
                self.podatki.append(
                    (date, time, location, operator, flightNr, route, type, registration, CnIn, aboard, fatalities,
                     ground, summary))
        self.podatki = np.sort(np.array(self.podatki, dtype=dtype), order='Date')

        koordinate = []
        with open("podatki\\koordinate.csv", 'r', encoding='utf-8') as file:
            csv_reader = DictReader(file)
            for row in csv_reader:
                location = row['Location Name']
                latitude = float(row["Latitude"]) if row["Latitude"] else 0.0
                longitude = float(row["Longitude"]) if row["Longitude"] else 0.0
                koordinate.append((location, latitude, longitude))

        self.crash_data = pd.DataFrame({
            'Location Name': [koordinata[0] for koordinata in koordinate],
            'Latitude': [float(koordinata[1]) for koordinata in koordinate],
            'Longitude': [float(koordinata[2]) for koordinata in koordinate]
        })
        self.years = np.unique([entry['Date'][-4:] for entry in self.podatki])
        self.years_numeric = np.array([int(year) for year in self.years])
        self.num_accidents = np.array(
            [np.sum([1 for entry in self.podatki if entry['Date'].endswith(year)]) for year in self.years],
            dtype=np.float64)
        self.num_fatalities = np.array(
            [np.sum([entry['Fatalities'] for entry in self.podatki if entry['Date'].endswith(year)]) for year in
             self.years], dtype=np.float64)
        self.num_passengers = np.array(
            [np.sum([entry['Aboard'] for entry in self.podatki if entry['Date'].endswith(year)]) for year in
             self.years], dtype=np.float64)
        self.ratio = self.num_accidents / self.num_passengers * 100

    def get_geolocations(self):
        # ZEMLJEVID SVETA - Pridobivanje koordinat in shranjevanje v .csv datoteko
        # !! PROGRAM TEČE OKOLI 3 URE !! PODATKI SO ŽE V CSV DATOTEKI !!
        """
        base_url = "https://nominatim.openstreetmap.org/search.php"
        lokacije = np.array([crash[2] for crash in podatki])
        koordinate = []
        for lokacija in lokacije:
            time.sleep(2)
            params = {
                'q': lokacija,
                'format': 'json'
            }
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    print("Display Name:", data[0]['display_name'])
                    print("Latitude:", data[0]['lat'])
                    print("Longitude:", data[0]['lon'])
                    koordinate.append(data[0])
                else:
                    print("No results found")

            else:
                print("Failed to fetch data")

        crash_data = pd.DataFrame({
            'Location Name': [koordinata['display_name'] for koordinata in koordinate],
            'Latitude': [float(koordinata['lat']) for koordinata in koordinate],
            'Longitude': [float(koordinata['lon']) for koordinata in koordinate]
        })
        # Shrani DataFrame v .csv datoteko
        crash_data.to_csv('podatki/koordinate.csv', index=False)
        """

    def crashes_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.num_accidents, marker='o', linestyle='-')
        plt.title('Number of Airplane Accidents Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.grid(True)
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def crashes_over_time_regression(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.years_numeric, self.num_accidents, marker='o')

        p = np.poly1d(np.polyfit(self.years_numeric, self.num_accidents, 3))
        plt.plot(self.years_numeric, p(self.years_numeric))
        plt.title('Number of Airplane Accidents Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.grid(True)
        plt.xticks(self.years_numeric[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def ratio_over_time(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.ratio, marker='o', linestyle='-')
        plt.title('Number of accidents / Total amount of passegers Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Ratio (%)')
        plt.grid(True)
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def ratio_over_time_regression(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.years_numeric, self.ratio, marker='o')

        p = np.poly1d(np.polyfit(self.years_numeric, self.ratio, 3))
        plt.plot(self.years_numeric, p(self.years_numeric))
        plt.title('Number of accidents / Total amount of passegers Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Ratio (%)')
        plt.grid(True)
        plt.xticks(self.years_numeric[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def fatilities_vs_ratio(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.subplots()
        ax1.plot(self.years, self.ratio, color='orange', marker=".", linewidth=1)
        ax1.set_xlabel('Year', fontsize=11)
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.set_ylabel('Ratio (%)', color='orange', fontsize=11)
        ax1.tick_params('y', colors='orange')
        ax2 = ax1.twinx()
        ax2.plot(self.years, self.num_accidents, color='red', marker=".", linewidth=1)
        ax2.set_ylabel('Number of fatalities', color='red', fontsize=11)
        ax2.tick_params('y', colors='r')
        plt.title('Accidents VS Ratio by Year', loc='Center', fontsize=14)

        ax1.set_xticks(self.years[::5])
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(True)
        for label in ax1.xaxis.get_ticklabels()[1::5]:
            label.set_visible(False)

        fig.tight_layout()
        plt.show()
    
    def deaths_over_time(self):
        years = np.array([int(entry[0][-4:]) for entry in self.podatki])  # Extract years from the date
        fatalities = np.array([entry[10] for entry in self.podatki])  # Extract fatalities data

        plt.figure(figsize=(18, 8))
        plt.plot(fatalities, years, 'o')
        plt.xlabel('Fatalities')
        plt.ylabel('Years')
        plt.show()

    def ratio_between_aboard_fatal(self):
        aboard = np.array([entry[9] for entry in self.podatki])
        fatalities = np.array([entry[10] for entry in self.podatki])

        survivors = aboard - fatalities

        plt.figure(figsize=(8, 8))
        labels = ['Survivors', 'Fatalities']
        sizes = [np.sum(survivors), np.sum(fatalities)]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Proportion of Survivors to Fatalities')
        plt.axis('equal') 
        plt.show()

    def fatilities_vs_ratio_regression(self):
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.subplots()

        plt.scatter(self.years_numeric, self.ratio, marker='o',color='orange')
        p = np.poly1d(np.polyfit(self.years_numeric, self.ratio, 3))
        plt.plot(self.years_numeric, p(self.years_numeric),color='orange')
        ax1.set_xlabel('Year', fontsize=11)
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        ax1.set_ylabel('Ratio (%)', color='orange', fontsize=11)
        ax1.tick_params('y', colors='orange')
        ax2 = ax1.twinx()

        plt.scatter(self.years_numeric, self.num_accidents, marker='o', color='red')
        p = np.poly1d(np.polyfit(self.years_numeric, self.num_accidents, 3))
        plt.plot(self.years_numeric, p(self.years_numeric), color='red')
        ax2.set_ylabel('Number of fatalities', color='red', fontsize=11)
        ax2.tick_params('y', colors='r')
        plt.title('Accidents VS Ratio by Year', loc='Center', fontsize=14)

        ax1.set_xticks(self.years_numeric[::5])
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(True)
        for label in ax1.xaxis.get_ticklabels()[1::5]:
            label.set_visible(False)

        fig.tight_layout()
        plt.show()

    def highest_crash_counts(self):
        registrations = np.array([crash[6] for crash in self.podatki])

        unique_registrations, counts = np.unique(registrations, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        top_10_indices = sorted_indices[:10]
        top_10_airplanes = unique_registrations[top_10_indices]

        top_10_dict = {}

        for airplane in top_10_airplanes:
            count = counts[np.where(unique_registrations == airplane)[0][0]]
            print(airplane, ":", count)
            top_10_dict[airplane] = count

        airplanes = list(top_10_dict.keys())
        crash_counts = list(top_10_dict.values())

        plt.figure(figsize=(10, 6))
        plt.bar(airplanes, crash_counts, color='skyblue')
        plt.title('Top 10 Airplanes with Highest Crash Counts')
        plt.xlabel('Airplane Registration Number')
        plt.ylabel('Number of Crashes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def extract_data_from_summary(self, include, exclude):
        extracted = np.array([entry for entry in self.podatki if
                              any(inc in entry["Summary"].lower() for inc in include)
                              and not
                              any(exc in entry["Summary"].lower() for exc in exclude)])
        return extracted

    def operator_performance(self):
        # Izračunamo izraz (število žrtev * število potnikov) - število letal za vsako letalsko podjetje
        podjetja = np.unique([entry['Operator'] for entry in self.podatki])
        uspešnost = []
        for podjetje in podjetja:
            nesreče_podjetja = [entry for entry in self.podatki if entry['Operator'] == podjetje]
            število_letov = np.sum([1 for entry in self.podatki if entry['Operator'] == podjetje])
            if (število_letov > 5):
                skupno_mrtvih = np.sum([entry['Fatalities'] for entry in nesreče_podjetja])
                skupno_potnikov = np.sum([entry['Aboard'] for entry in nesreče_podjetja])

                izraz = (skupno_mrtvih * skupno_potnikov) / število_letov
                uspešnost.append(izraz)

        max_uspešnost = max(uspešnost)
        uspešnost = [x / max_uspešnost for x in uspešnost]

        # Pridobimo top 25 in lowest 25 letalskih podjetij glede na izraz
        sorted_indices_top = np.argsort(uspešnost)[::-1][:25]
        sorted_indices_lowest = np.argsort(uspešnost)[:25]
        top_podjetja = podjetja[sorted_indices_top]
        lowest_podjetja = podjetja[sorted_indices_lowest]
        top_uspešnost = np.array(uspešnost)[sorted_indices_top]
        lowest_uspešnost = np.array(uspešnost)[sorted_indices_lowest]

        # Priprava stolpičnih grafov
        plt.figure(figsize=(12, 8))

        # Top 25 letalskih podjetij
        plt.subplot(2, 1, 1)
        plt.bar(top_podjetja, top_uspešnost, color='skyblue')
        plt.xlabel('Letalsko podjetje')
        plt.ylabel('Uspešnost')
        plt.title('Najboljših 25 letalskih podjetij glede na uspešnost')
        plt.xticks(rotation=90)

        # Lowest 25 letalskih podjetij
        plt.subplot(2, 1, 2)
        plt.bar(lowest_podjetja, lowest_uspešnost, color='salmon')
        plt.xlabel('Letalsko podjetje')
        plt.ylabel('Uspešnost')
        plt.title('Najslabših 25 letalskih podjetij glede na uspešnost')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    def crash_locations_on_map(self):
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()

        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)

        plt.scatter(self.crash_data['Longitude'], self.crash_data['Latitude'], color='red', marker='o',
                    label='nesreče po svetu',
                    s=1)
        plt.legend()
        plt.title('Letalske nesreče po svetu')
        plt.show()

    def passanger_and_fatalities_over_time(self):

        plt.figure(figsize=(10, 6))
        plt.bar(self.years, self.num_passengers, color='black')
        plt.bar(self.years, self.num_passengers - self.num_fatalities,
                color=[(np.random.random(), np.random.random(), np.random.random()) for _ in range(len(self.years))])
        plt.title('Passangers and Fatalities over time')
        plt.xlabel('Year')
        plt.ylabel('Number of Accidents')
        plt.xticks(self.years[::5], rotation=45)
        plt.tight_layout()
        plt.show()

    def wordcloud(self, field):
        text = str(self.podatki[field])
        plane_mask = np.array(Image.open('assets/airplane_mask.jpg'))

        stopwords = set(STOPWORDS)

        wc = WordCloud(background_color="white", max_words=4000, mask=plane_mask,
                       stopwords=stopwords)
        wc.generate(text)

        plt.figure(figsize=(10, 10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(field, loc='Center', fontsize=14)
        plt.show()

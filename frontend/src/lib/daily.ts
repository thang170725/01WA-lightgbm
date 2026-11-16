import type { ForecastPoint } from "@/routes/UploadApi";

export type DailyForecast = {
    date: string; // YYYY-MM-DD of the first hour in the block
    weekday: string; // Mon, Tue, ...
    label: string; // "Mon 09/01"
    start: string; // ISO time of first hour
    end: string; // ISO time of last hour
    total: number; // sum of 24 hourly values (kWh)
    totalLo: number; // sum of hourly lower bounds (conservative 80% band)
    totalHi: number;
    err: [number, number]; // [total-totalLo, totalHi-total] cho error bar
    average: number;
    peak: number;
    peakTime: string; // HH:mm
    min: number;
    minTime: string; // HH:mm
    hours: number; // number of hourly points in the block
};

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function weekdayOf(iso: string) {
    const [y, m, d] = iso.slice(0, 10).split("-").map(Number);
    return WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
}

/**
 * Gom dự báo theo giờ thành từng ngày: mỗi ngày = 24 giờ liên tiếp
 * kể từ giờ đầu tiên được dự báo (giờ ngay sau điểm cuối của dữ liệu lịch sử).
 */
export function toDailyForecast(hourly: ForecastPoint[]): DailyForecast[] {
    const days: DailyForecast[] = [];

    for (let i = 0; i < hourly.length; i += 24) {
        const block = hourly.slice(i, i + 24);
        if (block.length === 0) break;

        let peakPoint = block[0];
        let minPoint = block[0];
        let total = 0;
        let totalLo = 0;
        let totalHi = 0;

        for (const p of block) {
            total += p.value;
            totalLo += p.lo;
            totalHi += p.hi;
            if (p.value > peakPoint.value) peakPoint = p;
            if (p.value < minPoint.value) minPoint = p;
        }

        const first = block[0].time;
        const weekday = weekdayOf(first);

        days.push({
            date: first.slice(0, 10),
            weekday,
            label: `${weekday} ${first.slice(8, 10)}/${first.slice(5, 7)}`,
            start: first,
            end: block[block.length - 1].time,
            total,
            totalLo,
            totalHi,
            err: [total - totalLo, totalHi - total],
            average: total / block.length,
            peak: peakPoint.value,
            peakTime: peakPoint.time.slice(11, 16),
            min: minPoint.value,
            minTime: minPoint.time.slice(11, 16),
            hours: block.length,
        });
    }

    return days;
}

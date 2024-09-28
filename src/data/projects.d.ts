interface ProjectDescription {
  name: string;
  date: string | Date;
  description: string;
  link: string;
}

const projects : ProjectDescription[] = [
  {
    name: "Epileptic Uncertainty Framework",
    date: "08/08/2024",
    description: "Applied Heteroskedastic Regression to Seizure Prediction using a CNN. Acheive parity with comparable CNN model trained using the classical prediction framework.",
    link: "posts/euf-technical-report.html"
  },
]

export {ProjectDescription, projects}
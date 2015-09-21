#!/usr/bin/env python


class WorkItem(Object):
    # define an object to describe particular work items from job history:
    title           : String,
    company         : String,
    work_location   : String,
    start_date      : datetime.date(mmyyyy),
    end_date        : datetime.date(mmyyyy),
    description     : List[String]
    comment         : List[CommentItem]

class SchoolItem(Object):
    # define school and education details
    school      : String,
    degree      : String,
    program     : String,
    grad_date   : datetime.date(mmyyyy),
    locale      : String
    comment     : List[CommentItem]

class CommentItem(Object):
    # define comments that can be assigned to specific sections of a resume like post-it notes
    _id         : Object(String),
    section     : String,
    contributor : String,
    severity    : int,
    comment     : List[String],
    
class AccoladeItem(Object):
    # define a container for achievements to be listed on a resume detailed at the year level
    description : String,
    granted_dt  : datetime.date(yyyy)
    
class ResumeDataItem(Object):
    # define the fields for your resume here:
    _id         : Object(String),
    author      : String,
    revision    : String,
    name        : String,
    email       : String,
    phone       : String,
    headline    : String,
    locale      : String,
    summary     : String,
    experience  : List[WorkItem],
    education   : List[SchoolItem],    
    skills      : List[String],
    honors      : List[AccoladeItem],
    awards      : List[AccoladeItem],
    publication : List[AccoladeItem],
    comment     : List[CommentItem]
